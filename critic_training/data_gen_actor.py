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




# Normalizes a value to a 0-1 scale based on a minimum and maximum value
def min_max_norm(val, v_min=1000.0, v_max=2000.0):
    return (val - v_min) / (v_max - v_min)


# Inverts a normalized value back to its original scale
def invert_min_max_norm(val, v_min=1000.0, v_max=2000.0):
    return (val * (v_max - v_min)) + v_min


# Gathers a list of image file paths in sequences from subdirectories of a root folder
def get_sample_series_list(root_folder, sequence_size=13,
                           offset_start=0, shuffle_series=True,
                           random_state=None, interval=1, ends_with="*_bw.png"):
                           
    samples = []  # simple array to append all the entries present in the .csv file
    sub_folders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    sequence = []

    for folder in sub_folders:
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


# Organizes all image files in subfolders under the root folder into sequences
def get_sequence_samples(root_folder, sequence_size=13,
                            offset_start=0, shuffle_series=True,
                            random_state=None, interval=1):
    
    # Start with a list of sequential image series.
    samples = get_sample_series_list(root_folder=root_folder,
                                     sequence_size=sequence_size,
                                     offset_start=offset_start,
                                     shuffle_series=shuffle_series,
                                     random_state=random_state,
                                     interval=interval)

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
                    image = cv2.imread(batch_sample, cv2.IMREAD_GRAYSCALE)
                    images.append(image)
                    labels.append([0])

                except Exception as e:
                    print(f" [EXCEPTION ENCOUNTERED: {e}; skipping sample {batch_sample}.] ")

            # Convert images into numpy arrays
            x_train = np.array(images)

            # make y_train an array of 0
            y_train = np.array(labels)

            # Here we do not hold the values of X_train and y_train,
            # instead we yield the values.
            yield x_train, y_train