"""
rover_data_processor.py

# CS-472
# Documentation: The following webpages were referenced for information on various python functions
# for realsense frame number info - https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.frame.html#pyrealsense2.frame.get_frame_number
# for datetime info - https://stackoverflow.com/questions/7999935/python-datetime-to-string-without-microsecond-component
# for how to add a header to a csv file - https://stackoverflow.com/questions/20347766/pythonically-add-header-to-a-csv-file
# for using random integers - https://www.w3schools.com/python/ref_random_randint.asp
# for finding an element in a list of dictionaries - https://www.geeksforgeeks.org/check-if-element-exists-in-list-in-python/
"""
import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import time
import csv
import os
from imutils.video import FPS

# Paths for source and destination data
SOURCE_PATH = "/media/usafa/data/rover_data"
DEST_PATH = "/media/usafa/data/rover_data_processed"

# Parameters for image processing
# Define the range of white values to be considered for binary conversion
white_L, white_H = 200, 255
# Resize dimensions (quarter of 640x480)
resize_W, resize_H = 160, 120
# Crop dimensions to focus on relevant parts of the image
crop_W, crop_B, crop_T = 160, 120, 40  # Crop from top third down

def load_telem_file(path):
    """
    Loads telemetry data from a CSV file to associate video frames with sensor data like throttle and steering.
    """
    with open(path, "r") as f:
        dict_reader = csv.DictReader(f)
        return list(dict_reader)

def process_bag_file(source_file, dest_folder=None, skip_if_exists=True):
    """
    Processes a single .bag file, extracting frames and converting them to grayscale and binary images.
    Saves these images to a specified destination directory.
    """
    try:
        print(f"Processing {source_file}...")
        file_name = os.path.basename(source_file.replace(".bag", ""))
        dest_path = os.path.join(dest_folder or DEST_PATH, file_name)
        
        if skip_if_exists and os.path.isdir(dest_path):
            print(f"{file_name} previously processed; skipping.")
            return

        os.makedirs(dest_path, exist_ok=True)
        frm_lookup = load_telem_file(source_file.replace(".bag", ".csv"))

        # Setup RealSense pipeline
        config, pipeline = rs.config(), rs.pipeline()
        rs.config.enable_device_from_file(config, source_file, False)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        
        # Allow time for pipeline to stabilize
        time.sleep(1)
        playback = pipeline.get_active_profile().get_device().as_playback()
        playback.set_real_time(True)
        alignedFs = rs.align(rs.stream.color)
        fps = FPS().start()

        # Processing loop
        while playback.current_status() == rs.playback_status.playing:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                aligned_frames = alignedFs.process(frames)
                color_frame = aligned_frames.get_color_frame()
                
                # Skip if no telemetry data for frame
                frm_num = color_frame.frame_number
                result = [entry for entry in frm_lookup if entry["index"] == str(frm_num)]
                if not result: continue
                
                # Extract throttle, steering, and heading data
                throttle, steering, heading = result[0]["throttle"], result[0]["steering"], result[0]["heading"]
                color_frame = np.asanyarray(color_frame.get_data())

                # Image processing
                color_frame = cv2.resize(color_frame, (resize_W, resize_H))
                gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
                BW_frame = cv2.inRange(gray_frame, white_L, white_H)
                BW_frame = BW_frame[crop_T:crop_B, 0:crop_W]
                
                # Save processed images
                bw_frm_name = f"{int(frm_num):09d}_{throttle}_{steering}_{heading}_bw.png"
                cv2.imwrite(os.path.join(dest_path, bw_frm_name), BW_frame)
                fps.update()

            except Exception as e:
                print(e)
                continue
    except Exception as e:
        print(e)
    finally:
        # Cleanup and stats
        if fps: fps.stop()
        if playback and playback.current_status() == rs.playback_status.playing:
            playback.pause()
            if pipeline: pipeline.stop()
        print(f"Finished {source_file}. FPS: {fps.fps() if fps else 'N/A'}")

def main():
    """
    Main function to process all .bag files in the source directory.
    """
    for filename in filter(lambda f: f.endswith(".bag"), os.listdir(SOURCE_PATH)):
        process_bag_file(os.path.join(SOURCE_PATH, filename))

if __name__ == "__main__":
    main()
