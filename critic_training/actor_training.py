"""
model_training.py

# CS-472
# Documentation: The following webpages were referenced for information on various python functions
# for realsense frame number info - https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.frame.html#pyrealsense2.frame.get_frame_number
# for datetime info - https://stackoverflow.com/questions/7999935/python-datetime-to-string-without-microsecond-component
# for how to add a header to a csv file - https://stackoverflow.com/questions/20347766/pythonically-add-header-to-a-csv-file
# for using random integers - https://www.w3schools.com/python/ref_random_randint.asp
# for finding an element in a list of dictionaries - https://www.geeksforgeeks.org/check-if-element-exists-in-list-in-python/
"""

import data_gen_actor as data_gen
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout, Concatenate, Input
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot
import keras
import numpy as np

# Configuration parameters
DEVICE = "/GPU:0"  # Device to use for computation. Change to "/GPU:0" if GPU is available
#DATA_PATH = "/media/usafa/data/rover_data/processed"  # Path to the processed data
#DATA_PATH += "/smooth/left/g/"
DATA_PATH_FIX = "/media/usafa/data/rover_data/processed/take_2/corner_fix"
DATA_PATH_L = "/media/usafa/data/rover_data/processed/take_2/left"  # Path to the processed data
DATA_PATH_R = "/media/usafa/data/rover_data/processed/take_2/right"  # Path to the processed data
DATA_PATH_JL = "/media/usafa/data/rover_data/processed/smooth/left"  # Path to the processed data
DATA_PATH_JR = "/media/usafa/data/rover_data/processed/smooth/right"  # Path to the processed data
MODEL_NUM = 1 # Model number for naming
TRAINING_VER = 1  # Training version for naming
NUM_EPOCHS = 50  # Number of epochs to train
BATCH_SIZE = 13  # Batch size for training
TRAIN_VAL_SPLIT = 0.8  # Train/validation split ratio
MODEL_NAME = "critic_model_01_ver01_epoch0007_val_loss0.0007.h5"
MODEL_PATH = '/media/usafa/data/models/actors/'
CRITIC_PATH = '/media/usafa/data/models/critics/' + MODEL_NAME


# Define the CNN model structure
def define_model(input_shape=(120, 200, 1)):
    
    model = Sequential([
        
        Conv2D(32, (2, 2), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (2, 2), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (2, 2), activation='relu'),
        MaxPooling2D((3, 3)),

        Flatten(),

        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        
        Dense(2, activation=None)
    ])

    return model

def get_model(filename):
    """Load and compile the TensorFlow Keras model."""
    model = keras.models.load_model(filename, compile=False)
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    print("Loaded Model")
    return model


# Train the model with data from a generator, using checkpoints and a specified device
def train_model(amt_data=1.0):
    
    samples_fix = data_gen.get_sequence_samples(DATA_PATH_FIX, sequence_size=BATCH_SIZE)
    samples_l = data_gen.get_sequence_samples(DATA_PATH_L, sequence_size=BATCH_SIZE)
    samples_r = data_gen.get_sequence_samples(DATA_PATH_R, sequence_size=BATCH_SIZE)
    samples_jl = data_gen.get_sequence_samples(DATA_PATH_JL, sequence_size=BATCH_SIZE)
    samples_jr = data_gen.get_sequence_samples(DATA_PATH_JR, sequence_size=BATCH_SIZE)
    samples = samples_l + samples_r + samples_fix + samples_jl + samples_jr
    
    # You may wish to do simple testing using only 
    # a fraction of your training data...
    if amt_data < 1.0:
        # Use only a portion of the entire dataset
        samples, _\
            = data_gen.split_samples(samples, fraction=amt_data)

    # Now, split our samples into training and validation sets
    # Note that train_samples will contain a flat list of sequenced 
    # image file paths.
    train_samples, val_samples = data_gen.split_samples(samples, fraction=TRAIN_VAL_SPLIT)

    train_steps = int(len(train_samples) / BATCH_SIZE)
    val_steps = int(len(val_samples) / BATCH_SIZE)

    # Create data generators that will supply both the training and validation data during training.
    train_gen = data_gen.batch_generator(train_samples, batch_size=BATCH_SIZE, normalize_labels=True)
    val_gen = data_gen.batch_generator(val_samples, batch_size=BATCH_SIZE, normalize_labels=True)
    
    with tf.device(DEVICE):

        # Load the pretrained critic model from memory
        critic_model = keras.models.load_model(CRITIC_PATH)

        # Freeze the weights of the pretrained critic
        critic_model.trainable = False

        # Define your actor model
        shape = (120, 200, 1)  # Define your input shape
        actor_model = define_model(input_shape=shape)

        # Connect the actor model to the pretrained critic
        # Combine the actor and the pretrained critic
        actor_input_image = actor_model.input
        critic_output = critic_model([actor_input_image, actor_model.output])
        model = Model(inputs=actor_input_image, outputs=critic_output)

        # Compile the combined model
        model.compile(optimizer='adam', loss='mse')

        model.summary()  # Print a summary of the model architecture
        
        # Path for saving the best model checkpoints
        filePath = MODEL_PATH + "actor_model_" + f"{MODEL_NUM:02d}_ver{TRAINING_VER:02d}" + "_epoch{epoch:04d}_val_loss{val_loss:.4f}.h5"
        
        # Save only the best (i.e. min validation loss) epochs
        checkpoint_best = ModelCheckpoint(filePath, monitor="val_loss", 
                                          verbose=1, save_best_only=True, 
                                          mode="min")
        
        # Train your model here.
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=NUM_EPOCHS,
            verbose=1,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=[checkpoint_best]
        )

    return history


# Plot training and validation loss over epochs
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(len(histories),1,1)
        pyplot.title('Training Loss Curves')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')

    pyplot.show()

# Run the training process and display training diagnostics
def main():

    # amt_data : proportion of data to use for training/testing
    history = train_model(amt_data=0.0001)
    summarize_diagnostics([history])


# Entry point to start the training process
if __name__ == "__main__":
    main()
