# Documentation is in rover_recorder.py

import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import time
import csv
import os
from imutils.video import FPS

#set source and destination paths
SOURCE_PATH = '/home/usafa/usafa_472/rover_lab_01/data/unprocessed/'
DEST_PATH = '/home/usafa/usafa_472/rover_lab_01/data/processed/'
#define the range of white you want
white_L = 220
white_H = 255

#resizing parameters
resize_W = 320
resize_H = 240

#cropping parameters origin is top left of image so crop_B is the full height of the image (the bottom row visually) and crop_T is the highest part of the image we want to see but the lowest numeric row of pixels we want to see.
crop_W = int(resize_W)
crop_B = resize_H
crop_T = int(resize_H/3)


def load_telem_file(path):
    # Create lookup for frame index (ID)
    #this is a list of dictionaries
    with open(path, 'r') as f:
        # Load data from the data file (comma delimited)
        dict_reader = csv.DictReader(f)
        # hold it in a structure for quick lookup
        list_of_dict = list(dict_reader)

        return list_of_dict


def process_bag_file(source_file, dest_folder=None, skip_if_exists=True):
    fps = None
    pipeline = None

    try:
        i = 0
        print(f"Processing {source_file}...")
        # EX: .../20210122-120614.bag
        path = source_file

        # EX: .../20210122-120614
        file_name = os.path.basename(path.replace(".bag", ""))
        #set destination
        if dest_folder is None:
            dest_path = os.path.join(DEST_PATH, file_name)
        else:
            dest_path = os.path.join(dest_folder, file_name)

        #skip over previously processed files
        if skip_if_exists:
            if os.path.isdir(dest_path):
                print(f"{file_name} was previously processed; skipping file...")
                return

        # Make subfolder to hold all training data
            # File: '.../20210122-120614' now exists
        os.makedirs(dest_path, exist_ok=True)

        # Load data associated with the video.
        # frm_lookup is [{'index', 'throttle', 'steering', 'heading'} ...]
        frm_lookup = load_telem_file(path.replace(".bag", ".csv"))

        #set up stream from bag
        config = rs.config()

        rs.config.enable_device_from_file(config, path, False)
        pipeline = rs.pipeline()

        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        time.sleep(1)

        # Pause here to allow pipeline to start
        # before turning off real-time streaming.
        profile = pipeline.get_active_profile()

        playback = profile.get_device().as_playback()
        playback.set_real_time(True)

        align_to = rs.stream.color
        alignedFs = rs.align(align_to)
        fps = FPS().start()
        #loop until all frames are done
        while playback.current_status() == rs.playback_status.playing:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                if not frames:
                    print("no frames")
                    continue

                # align rgb to depth pixels
                aligned_frames = alignedFs.process(frames)

                color_frame = aligned_frames.get_color_frame()

                # Match frame/image to control data--------------------------

                # Get related throttle and steering for frame
                frm_num = color_frame.frame_number
                #check if there is data recorded for the given frame
                result = list(filter(lambda entry: entry['index'] == str(frm_num), frm_lookup))
                #if no data is available restart loop
                if len(result) == 0:
                    continue
                #extract the data corresponding to the current frame
                throttle = result[0]['throttle']
                steering = result[0]['steering']
                heading = result[0]['heading']

                color_frame = np.asanyarray(color_frame.get_data())

                # resize frame
                color_frame = cv2.resize(color_frame, (resize_W, resize_H))

                #convert to BW image for easier line detection
                gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
                BW_frame = cv2.inRange(gray_frame, white_L, white_H)

                # Crop Bw image
                BW_frame = BW_frame[crop_T:crop_B, 0:crop_W]

                i += 1

                # show the output frame for sanity check
                cv2.imshow("gray", gray_frame)
                cv2.imshow("Black and white", BW_frame)
                cv2.imshow("Color Processed", color_frame)


                # rgb
                c_frm_name = f"{'{:09d}'.format(frm_num)}_{throttle}_{steering}_{heading}_c.png"

                # BW
                bw_frm_name = f"{'{:09d}'.format(frm_num)}_{throttle}_{steering}_{heading}_bw.png"

                #save both images
                cv2.imwrite(os.path.join(dest_path, c_frm_name), color_frame)
                cv2.imwrite(os.path.join(dest_path, bw_frm_name), BW_frame)

                #keep track of fps
                fps.update()

                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break
            except Exception as e:
                print(e)
                continue
    except Exception as e:
        print(e)
    finally:
        pass
#checks to end loop
    try:

        # stop recording
        if fps is not None:
            fps.stop()
        time.sleep(0.5)
        if playback is not None:
            if playback.current_status() == rs.playback_status.playing:
                playback.pause()
                if pipeline is not None:
                    pipeline.stop()
                    time.sleep(0.5)
    except Exception as e:
        print("Unexpected error during cleanup.", exc_info=True)

#finishing messages
    print(f"Finished processing frames for {source_file}.")
    if fps is not None:
        print("Elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


def main():
    #loop through bag files in the given directory
    for filename in os.listdir(SOURCE_PATH):
        if filename.endswith(".bag"):
            source_file = os.path.join(SOURCE_PATH, filename)
            process_bag_file(source_file)
        else:
            continue


if __name__ == "__main__":
    # for earlier datasets... not needed for new data
    # tweak_data_samples("training_data/McCurdy")

    main()
