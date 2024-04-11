"""
rover_driver.py

# CS-472
# Documentation: The following webpages were referenced for information on various python functions
# for realsense frame number info - https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.frame.html#pyrealsense2.frame.get_frame_number
# for datetime info - https://stackoverflow.com/questions/7999935/python-datetime-to-string-without-microsecond-component
# for how to add a header to a csv file - https://stackoverflow.com/questions/20347766/pythonically-add-header-to-a-csv-file
# for using random integers - https://www.w3schools.com/python/ref_random_randint.asp
# for finding an element in a list of dictionaries - https://www.geeksforgeeks.org/check-if-element-exists-in-list-in-python/
"""

import pyrealsense2.pyrealsense2 as rs
import time
import numpy as np
import cv2
import keras
import utilities.drone_lib as dl
import os
import tensorflow as tf
from keras.optimizers import Adam
from random import shuffle
import glob
from tensorflow.keras.losses import Loss
from keras.callbacks import ModelCheckpoint

# Path to the trained model weights
MODEL_NAME = "models/rover_model_01_ver01_epoch0012_val_loss0.0015.h5"
DEFAULT_PORT = "/dev/ttyACM0"
DEVICE = "/GPU:0"

# Rover driving command limits
MIN_THROTTLE, MAX_THROTTLE = 982, 2006
MIN_STEERING, MAX_STEERING = 982, 2002


# Image processing parameters
white_L, white_H = 200, 255  # White color range
DEST_PATH = '/media/usafa/data/rover_data/processed' + '/rl/'
MODEL_PATH = '/media/usafa/data/models' + '/rl/'


#resizing parameters
resize_W = 200
resize_H = 150

#cropping parameters origin is top left of image so crop_B is the full height of the image (the bottom row visually) and crop_T is the highest part of the image we want to see but the lowest numeric row of pixels we want to see.
crop_W = int(resize_W)
crop_B = resize_H
crop_T = int(resize_H/5)

class RLLoss(Loss):
    def __init__(self):
        super(RLLoss, self).__init__()
        self.mtx = []
        for i in range(0, 120):
            self.mtx.append([])
            for j in range(0, 200):
                if i<120/4 or i>120-120/4:
                    pt = 0
                else:
                    pt = (abs(100-j)-50)/100
                self.mtx[i].append(pt)
        self.mtx = tf.cast(np.array(self.mtx), tf.float32)

    def call(self, y_true, y_pred):
        print("bfbyfuwibfuyi")
        y_true = tf.cast(y_true, tf.float32)
        return tf.reduce_sum(y_true * self.mtx)*y_pred[:, 0]

def batch_generator(samples, batch_size=13,
                    normalize_labels=True,
                    y_min=1000.0, y_max=2000.0):
    
    num_samples = len(samples)
    while True:
        for offset in range(0, num_samples, batch_size):
            # print(f" [Serving batch {offset} - {offset+batch_size}...] ")
            batch_samples = samples[offset:offset + batch_size]
            # Sanity check
            # print(f" {sample_name} data range: {offset}:{offset + batch_size}")
            images = []
            labels = []
            for batch_sample in batch_samples:
                try:
                    file_name = os.path.basename(batch_sample).replace(".png", "")
                    throttle = int(file_name.split('_')[1])
                    image = cv2.imread(batch_sample, cv2.IMREAD_GRAYSCALE)
                    images.append(image)

                    if normalize_labels:
                        throttle = min_max_norm(throttle, y_min, y_max)

                    labels.append(image)

                except Exception as e:
                    print(f" [EXCEPTION ENCOUNTERED: {e}; skipping sample {batch_sample}.] ")

            x_train = np.array(images)
            y_train = np.array(labels)
            yield x_train, y_train

def min_max_norm(val, v_min=1000.0, v_max=2000.0):
    return (val - v_min) / (v_max - v_min)

def get_sample_series_list(folder, sequence_size=13,
                           offset_start=0, shuffle_series=True,
                           random_state=None, interval=1, ends_with="*_bw.png"):
                           
    samples = []  # simple array to append all the entries present in the .csv file
    sequence = []


    path = os.path.join(folder, ends_with)
    files = sorted(glob.glob(path))
    file_count = 0
    for file in files:
        if file_count % interval == 0:
            if len(sequence) >= sequence_size:
                # Add sequence to list of sequences/samples
                samples.append(sequence)

                # reset sequence list
                sequence = []
            
            # continue to build current sequence...
            file_count += 1
            if file_count >= offset_start:
                sequence.append(file)

    if shuffle_series:
        # Shuffle the order of sequences so that they are not contiguous.
        shuffle(samples, random =random_state)

    return samples

def get_sequence_samples(root_folder, sequence_size=13,
                            offset_start=0, shuffle_series=True,
                            random_state=None, interval=1):
    
    # Start with a list of sequential image series.
    samples = get_sample_series_list(folder=root_folder,
                                     sequence_size=sequence_size,
                                     offset_start=offset_start,
                                     shuffle_series=shuffle_series,
                                     random_state=random_state,
                                     interval=interval)

    # Finally, convert this list of lists into a single flat list of images.
    samples = [item for sublist in samples for item in sublist]
    return samples

def get_model(filename):
    """Load and compile the TensorFlow Keras model."""
    model = keras.models.load_model(filename, compile=False)
    model.compile(optimizer=Adam(), loss=RLLoss())
    print("Loaded Model")
    return model

def min_max_norm(val, v_min=1000.0, v_max=2000.0):
    return (val - v_min) / (v_max - v_min)


def invert_min_max_norm(val, v_min=1000.0, v_max=2000.0):
    return (val * (v_max - v_min)) + v_min


def denormalize(steering, throttle):
    """Denormalize steering and throttle values to the rover's command range."""
    steering = invert_min_max_norm(steering, MIN_STEERING, MAX_STEERING)
    throttle = invert_min_max_norm(throttle, MIN_THROTTLE, MAX_THROTTLE)
    return steering, throttle

def initialize_pipeline():
    """Initialize the RealSense pipeline for video capture."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

def get_video_data(pipeline):
    """Capture a video frame, preprocess it, and prepare it for model prediction."""
    frame = pipeline.wait_for_frames()
    color_frame = frame.get_color_frame()
    if not color_frame:
        return None

    image = np.asanyarray(color_frame.get_data())

    # resize frame
    image = cv2.resize(image, (resize_W, resize_H))

    #convert to BW image for easier line detection
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.inRange(image, white_L, white_H)

    # Crop Bw image
    image = image[crop_T:crop_B, 0:crop_W]

    return image

def set_rover_data(rover, steering, throttle):
    """Set rover control commands, ensuring they're within the valid range."""
    
    # May uncomment below to force a specific range, if your model is 
    # sometimes outputting weird ranges (probably not needed)
    steering, throttle = check_inputs(int(steering), int(throttle))
    
    rover.channels.overrides = {"1": steering, "3": throttle}
    print(f"Steering: {steering}, Throttle: {throttle}")


def check_inputs(steering, throttle):
    """Check and clamp the steering and throttle inputs to their allowed ranges."""
    steering = np.clip(steering, MIN_STEERING, MAX_STEERING)
    throttle = np.clip(throttle, MIN_THROTTLE, MAX_THROTTLE)
    return steering, throttle

def train_model(model, session, batch_size=13):
    samples = get_sequence_samples(os.path.join(DEST_PATH, str(session)), sequence_size=batch_size)
    train_steps = int(len(samples) / 13)
    train_gen = batch_generator(samples, batch_size=batch_size, normalize_labels=True)
    
    
    # Path for saving the best model checkpoints
    model_dir = os.path.join(MODEL_PATH, f"rover_model_{session}.h5")
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)
    
    # Train your model here.
    # Save only the best (i.e. min validation loss) epochs
    checkpoint_best = ModelCheckpoint(model_dir, monitor="loss", 
                                        verbose=1, save_best_only=True, 
                                        mode="min")
    
    # Train your model here.
    model.fit(
        train_gen,
        epochs=1,
        verbose=1,
        steps_per_epoch=train_steps,
        callbacks=[checkpoint_best]
    )

def main():

    """Main function to drive the rover based on model predictions."""
   
    # Setup and connect to the rover
    rover = dl.connect_device(DEFAULT_PORT)

    # Load the trained model
    model = get_model(MODEL_NAME)

    if model is None:
        print("Unable to load CNN model!")
        rover.close()
        print("Terminating program...")
        exit()
        
    while True:
        print("Arm vehicle to start mission.")
        print("(CTRL-C to stop process)")
        while not rover.armed:
            time.sleep(1)
        
        session = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
        
        # Initialize video capture
        pipeline = initialize_pipeline()
        counter = 0

        try:
            os.mkdir(DEST_PATH + '/' + str(session))
        except FileExistsError:
            print("Session directory already exists.")
        
        while rover.armed:
            processed_image = get_video_data(pipeline)
            if processed_image is None:
                print("No image from camera.")
                continue
            new_processed_image = np.expand_dims(processed_image, axis=-1)
            new_processed_image = np.expand_dims(new_processed_image, axis=0)

            # Predict steering and throttle from the processed image
            output = model.predict(new_processed_image)
            steering, throttle = denormalize(output[0][0], output[0][1])
            steering, throttle = int(steering), int(throttle)

            # Send commands to the rover
            set_rover_data(rover, steering, throttle)

            # save img with values, dest path has timestamp
            frm_name = f"{'{:09d}'.format(counter)}_{throttle}_{steering}_bw.png"
            cv2.imwrite(os.path.join(DEST_PATH, str(session), frm_name), processed_image)
            counter += 1

        # stop recording
        pipeline.stop()
        time.sleep(1)
        pipeline = None
        print("Done recording. Proceeding with training...")

        train_model(model, session)

    rover.close()


if __name__ == "__main__":
    main()
