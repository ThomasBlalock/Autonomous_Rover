"""
data_gen.py

# CS-472
# Documentation: The following webpages were referenced for information on various python functions
# for realsense frame number info - https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.frame.html#pyrealsense2.frame.get_frame_number
# for datetime info - https://stackoverflow.com/questions/7999935/python-datetime-to-string-without-microsecond-component
# for how to add a header to a csv file - https://stackoverflow.com/questions/20347766/pythonically-add-header-to-a-csv-file
# for using random integers - https://www.w3schools.com/python/ref_random_randint.asp
# for finding an element in a list of dictionaries - https://www.geeksforgeeks.org/check-if-element-exists-in-list-in-python/
"""

import cv2
import os
import glob
from random import shuffle
import numpy as np

# Constants defining the range of steering and throttle values
STEERING_MIN = 982
STEERING_MAX = 2002
THROTTLE_MIN = 982
THROTTLE_MAX = 2006


# class Loss:

#     def __init__(self, w=200, h=120, steering_priority_ratio=2):

#         # Create a matrix with 1s in the center and 0s at the edges
#         # This matrix will dot with the image and be divided by the throttle
#         # If there are a lot of 1s in the center, the loss will be lower
#         self.th_mtx = []
#         for i in range(0, h):
#             self.th_mtx.append([])
#             for j in range(0, w):
#                 if i<w*3/4 and j>w/4 and j<w*3/4:
#                     pt = 1
#                 else:
#                     pt = 0
#                 self.th_mtx[i].append(pt)
#         self.th_mtx = np.array(self.th_mtx).astype(np.float32)

#         # create a matrix with a value increasing from the center to the edges
#         # This matrix will dot with the image and be summed
#         # If the image is more white in the center, the loss will be lower
#         self.ed_mtx = []
#         for i in range(0, h):
#             self.ed_mtx.append([])
#             for j in range(0, w):
#                 if not (i<w*3/4 and j>w/4 and j<w*3/4):
#                     pt = ((abs(w-j/2) / (w+1) + 1)*steering_priority_ratio) ** 2
#                 else:
#                     pt = 0
#                 self.ed_mtx[i].append(pt)
#         self.ed_mtx = np.array(self.ed_mtx).astype(np.float32)

#     def __call__(self, image_s, image_e, throttle):

#         # Image start (current image)
#         image_s = image_s.astype(np.float32)
#         image_s = image_s / np.sum(image_s)

#         # Image end (image after t time steps)
#         image_e = image_e.astype(np.float32)
#         image_e = image_e / np.sum(image_e)

#         # throttle loss - incentivise going fast with white tape in front of you
#         th_loss = np.sum(image_s * self.th_mtx) / throttle

#         # edge loss - incentivise getting to the middle of the track
#         ed_loss = np.sum(image_e * self.ed_mtx)

#         return th_loss + ed_loss


class Loss:

    def __init__(self, w=200, h=120, steering_priority_ratio=2):

        # Create a matrix with 1s in the center and 0s at the edges
        # This matrix will dot with the image and be multiplied by the throttle
        # If there are a lot of 1s in the center, the loss will be lower
        self.th_mtx = []
        for i in range(0, h):
            self.th_mtx.append([])
            for j in range(0, w):
                if i<w*3/4 and j>w/4 and j<w*3/4:
                    pt = 1
                else:
                    pt = 0
                self.th_mtx[i].append(pt)
        self.th_mtx = np.array(self.th_mtx).astype(np.float32)

        # create a matrix with a value increasing from the center to the edges
        # This matrix will dot with the image and be summed
        # If the image is more white in the center, the loss will be lower
        self.ed_mtx = []
        for i in range(0, h):
            self.ed_mtx.append([])
            for j in range(0, w):
                if j<w/4:
                    pt = 4*j/w
                elif j>w*3/4:
                    pt = 4*(w-j)/w
                else:
                    pt = 0
                self.ed_mtx[i].append(pt)
        self.ed_mtx = np.array(self.ed_mtx).astype(np.float32)

    def __call__(self, image_s, image_e, throttle):

        # Image start (current image)
        image_s = image_s.astype(np.float32)
        image_s = image_s / (np.sum(image_s)+1)

        # Image end (image after t time steps)
        image_e = image_e.astype(np.float32)
        image_e = image_e / (np.sum(image_e)+1)

        # throttle loss - incentivise going fast with white tape in front of you
        th = np.sum(image_s * self.th_mtx) * max(throttle, 1)

        # edge loss - incentivise getting to the middle of the track
        ed = np.sum(image_e * self.ed_mtx)

        score = 2 - th - ed

        # The score should never be less than 0 or greater than 2
        if score<0:
            print("Faulty Score: "+score+", th: "+th+", ed: "+ed)
            return 0
        elif score>2:
            print("Faulty Score: "+score+", th: "+th+", ed: "+ed)
            return 2
        else:
            return 2 - th - ed
    

# Normalizes a value to a 0-1 scale based on a minimum and maximum value
def min_max_norm(val, v_min=1000.0, v_max=2000.0):
    return (val - v_min) / (v_max - v_min)


# Inverts a normalized value back to its original scale
def invert_min_max_norm(val, v_min=1000.0, v_max=2000.0):
    return (val * (v_max - v_min)) + v_min


# Gathers a list of image file paths in sequences from subdirectories of a root folder
def get_sample_series_list(folder, sequence_size=13,
                           offset_start=0, shuffle_series=True,
                           random_state=None, interval=1, ends_with="*_bw.png",
                           time_steps=5):
                           
    samples = []  # simple array to append all the entries present in the .csv file
    sub_folders = [f.path for f in os.scandir(folder) if f.is_dir()]
    sequence = []

    for folder in sub_folders:
        path = os.path.join(folder, ends_with)
        files = sorted(glob.glob(path))
        file_count = 0
        for file in files:
            file_count += 1
            if file_count >= offset_start:
                sequence.append(file)
    
        # Each sample will include both the image and a future image
        batch = []
        for i, file in enumerate(sequence):
            if i % interval == 0:
                samples.append(batch)
                batch = []
            if i < len(sequence) - time_steps:
                batch.append( [file, sequence[i+time_steps]] )

    if shuffle_series:
        # Shuffle the order of sequences so that they are not contiguous.
        shuffle(samples, random =random_state)

    return samples


# Organizes all image files in subfolders under the root folder into sequences
def get_sequence_samples(root_folder, sequence_size=13,
                            offset_start=0, shuffle_series=True,
                            random_state=None, interval=1,
                            time_steps=5):
    
    # Start with a list of sequential image series.
    samples = get_sample_series_list(folder=root_folder,
                                     sequence_size=sequence_size,
                                     offset_start=offset_start,
                                     shuffle_series=shuffle_series,
                                     random_state=random_state,
                                     interval=interval, time_steps=time_steps)

    # Finally, convert this list of lists into a single flat list of images.
    samples = [item for sublist in samples for item in sublist]
    return samples


# Splits the samples into two sets based on a specified fraction
def split_samples(samples, fraction=0.8):
    length = len(samples)
    num_training = int(fraction * length)
    return samples[:num_training], samples[num_training:]


def batch_generator(samples, batch_size=13,
                    normalize_labels=True,
                    y_min=1000.0, y_max=2000.0):
    
    num_samples = len(samples)
    loss_fn = Loss(steering_priority_ratio=2)
    while True:
        for offset in range(0, num_samples, batch_size):
            # print(f" [Serving batch {offset} - {offset+batch_size}...] ")
            batch_samples = samples[offset:offset + batch_size]
            # Sanity check
            # print(f" {sample_name} data range: {offset}:{offset + batch_size}")
            images = []
            controls = []
            labels = []
            for batch_sample in batch_samples:
                try:
                    # s = start (current image), e = end (future image)
                    batch_sample_s = batch_sample[0]
                    batch_sample_e = batch_sample[1]
                    file_name_s = os.path.basename(batch_sample_s).replace(".png", "")

                    # IMPORTANT NOTE: Be sure that these fields line up 
                    # with your particular file naming convention!
                    throttle = file_name_s.split('_')[1]
                    steering = file_name_s.split('_')[2]

                    throttle = int(throttle)
                    throttle = min_max_norm(throttle, THROTTLE_MIN, THROTTLE_MAX)

                    steering = int(steering)
                    steering = min_max_norm(steering, STEERING_MIN, STEERING_MAX)

                    controls.append([throttle, steering])
                    
                    image_s = cv2.imread(batch_sample_s, cv2.IMREAD_GRAYSCALE)
                    images.append(image_s)

                    image_e = cv2.imread(batch_sample_e, cv2.IMREAD_GRAYSCALE)
                    score = loss_fn(image_s, image_e, throttle)

                    labels.append([score])

                except Exception as e:
                    print(f" [EXCEPTION ENCOUNTERED: {e}; skipping sample {batch_sample}.] ")

            # Convert into numpy arrays
            images = np.array(images)
            controls = np.array(controls)
            x_train = (images, controls)
            y_train = np.array(labels)

            # Here we do not hold the values of X_train and y_train,
            # instead we yield the values.
            yield x_train, y_train