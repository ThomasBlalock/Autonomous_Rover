# for realsense frame number info - https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.frame.html#pyrealsense2.frame.get_frame_number
# for datetime info - https://stackoverflow.com/questions/7999935/python-datetime-to-string-without-microsecond-component
# for how to add aheader to a csv file - https://stackoverflow.com/questions/20347766/pythonically-add-header-to-a-csv-file
# for using random integers - https://www.w3schools.com/python/ref_random_randint.asp
# for finding an element in a list of dictionaries - https://www.geeksforgeeks.org/check-if-element-exists-in-list-in-python/
# for converting to black and white image - https://techtutorialsx.com/2019/04/13/python-opencv-converting-image-to-black-and-white/
# for cropping images - https://learnopencv.com/cropping-an-image-using-opencv/


import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
from dronekit import connect, VehicleMode, LocationGlobalRelative
import datetime
import rosbag
import csv
import random
import time

DATA_FILEPATH = '/media/usafa/data/rover_data/unprocessed'
DATA_FILEPATH += '/test/'

#initialize camera settings and turn on camera
def initialize_pipeline(run):
    pipeline = rs.pipeline()
    config = rs.config()
    #record video to bag file
    config.enable_record_to_file(f'{DATA_FILEPATH}{run}.bag')
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

#use to connect to rover in future but not used right now
def connect_device(s_connection):
    print("Connecting to device...")
    device = connect(ip=s_connection, wait_ready=True)
    print("Device connected.")
    print(f"Device version: {device.version}")
    return device

#gets color image frame from stream, displays it, and returns the frame number for use in data organization
def get_video_data(pipeline):

    frame = pipeline.wait_for_frames() #get frame
    color_frame = frame.get_color_frame()

    if not color_frame: #if no frame is available signal to restart loop
        return None

    color_image = np.asanyarray(color_frame.get_data()) #get image from frame data
    color_image_fn = color_frame.get_frame_number()
    cv2.imshow('color', color_image) #display image

    return color_image_fn

#in future collect relavent data form the rover, right now return random ints for each field
def get_rover_data(rover):
    if (rover.channels['3'] is not None and
        rover.channels['1'] is not None):
        throttle = int(rover.channels['3'])
        steering = int(rover.channels['1'])
    else:
        return None
    heading = rover.heading

    return [throttle, steering, heading]

#add data points to the csv file
def append_data(data, index, data_file):
    field_names = ['index', 'throttle', 'steering', 'heading']
    data_dict = {'index': index, 'throttle': data[0], 'steering': data[1], 'heading': data[2]}
    csv.DictWriter(data_file,  fieldnames=field_names).writerow(data_dict)

def device_channel_msg(device):
    @device.on_message('RC_CHANNELS')
    def RC_CHANNEL_listener(vehicle, name, message):
        set_rc(vehicle, 1, message.chan1_raw)
        set_rc(vehicle, 2, message.chan2_raw)
        set_rc(vehicle, 3, message.chan3_raw)
        set_rc(vehicle, 4, message.chan4_raw)
        set_rc(vehicle, 5, message.chan5_raw)
        set_rc(vehicle, 6, message.chan6_raw)
        set_rc(vehicle, 7, message.chan7_raw)
        set_rc(vehicle, 8, message.chan8_raw)
        set_rc(vehicle, 9, message.chan9_raw)
        set_rc(vehicle, 10, message.chan10_raw)
        set_rc(vehicle, 11, message.chan11_raw)
        set_rc(vehicle, 12, message.chan12_raw)
        set_rc(vehicle, 13, message.chan13_raw)
        set_rc(vehicle, 14, message.chan14_raw)
        set_rc(vehicle, 15, message.chan15_raw)
        set_rc(vehicle, 16, message.chan16_raw)
        vehicle.notify_attribute_listeners('channels', vehicle.channels)
    
def set_rc(vehicle, chnum, v):
    vehicle._channels._update_channel(str(chnum), v)

def main():
    #get starting timestamp for file naming
    run = datetime.datetime.now()
    #create bag file
    bag = rosbag.Bag(f'{DATA_FILEPATH}{run}.csv', 'w')
    #create csv and add headers
    header = ['index', 'throttle', 'steering', 'heading']
    with open(f'{DATA_FILEPATH}{run}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    #reopen csv for the remainder of the run
    data_file = open(f'{DATA_FILEPATH}{run}.csv', 'a')

    pipeline = initialize_pipeline(run)
    rover = connect('/dev/ttyACM0', wait_ready=True, baud=57600)
    device_channel_msg(rover)

    print("Waiting for rover to be armed...")
    while not rover.armed:
        time.sleep(1)
    print("Rover armed.")

    while rover.armed:
        #get index of current frame
        # Image dimentions = [480, 640, 3]
        index = get_video_data(pipeline)
        if index == None:
            # skip the rest of the loop if no frame is available
            continue

        #get data from rover: throttle, steering, heading
        cntrl_data = get_rover_data(rover)
        if cntrl_data is None:
            continue
        #add data to csv with current frame index
        append_data(cntrl_data, index, data_file)


#exit condition
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

#close csv file
    data_file.close()
    print('done')


main()