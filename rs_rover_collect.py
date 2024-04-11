from imutils.video import FPS
from dronekit import connect
import argparse
import pyrealsense2.pyrealsense2 as rs
from pymavlink import mavutil
import time
import logging
import sys
import utilities.drone_lib as dl
import csv


# port = "/dev/ttyUSB0" #USB
DEFAULT_BAUD = 115200  # 57600
DEFAULT_DATA_PATH = '/media/usafa/data/rover_data/'
# DEFAULT_DATA_PATH += 'unprocessed/test/'
DEFAULT_DATA_PATH += 'unprocessed/take_2/corner_fix/'
# DEFAULT_DATA_PATH += 'unprocessed/off_course/'
# DEFAULT_DATA_PATH += 'unprocessed/const_throttle/left/'
DEFAULT_PORT = "/dev/ttyACM0"
connection = None

def append_ardu_data(throttle, steering, heading, idx, file):
    f = open(file, "a+")
    f.write(f"{throttle},{steering},{heading},{idx}\n")
    f.close()

def append_data(data, index, data_file):
    field_names = ['index', 'throttle', 'steering', 'heading']
    data_dict = {'index': index, 'throttle': data[0], 'steering': data[1], 'heading': data[2]}
    csv.DictWriter(data_file,  fieldnames=field_names).writerow(data_dict)

def prepare_log_file(log_file):
    pass
    # prepare log file...
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    #handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    #logging.basicConfig(level=logging.DEBUG, handlers=handlers)


def set_rc(vehicle, chnum, v):
    vehicle._channels._update_channel(str(chnum), v)


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


def collect_data(bag_file):
    throttle = 0
    steering_mix = 0
    state_update_interval = 30 # (update after every 30 frames)

    try:

        file_name = bag_file.replace(".bag", ".csv")
        data_file = open(file_name, 'w')

        header = ['index', 'throttle', 'steering', 'heading']
        writer = csv.writer(data_file)
        writer.writerow(header)

        logging.info(f"Recording to be stored in location: {bag_file}.")
        logging.info("Preparing RealSense data streams...")

        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_record_to_file(f"{bag_file}")

        logging.info("configuring rgb stream.")
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        logging.info("Starting camera streams...")
        pipeline.start(config)

        fps = FPS().start()
        logging.info("Recording for realsense sensor streams started.")

    except Exception as e:
        logging.error("Unexpected error.", exc_info=True)
        return

    # loop over the frames from the video stream...
    while connection.armed:
        try:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            bgr_frame = frames.get_color_frame()

            if not bgr_frame:
                continue

            # For Ardurover channel mapping:
            #    3 is throttle
            #    1 is steering

            # get data from device...
            if (not connection.channels['3'] is None
                    and not connection.channels['1'] is None):
                throttle = int(connection.channels['3'])
                steering_mix = int(connection.channels['1'])

            heading = connection.heading

            frm_num = int(bgr_frame.frame_number)

            # write throttle and steering related to current frame...
            #append_ardu_data(throttle=throttle, steering=steering_mix,heading=heading, idx=frm_num,file=ardu_file)

            data = [throttle, steering_mix, heading]
            append_data(data, frm_num, data_file)

            if (frm_num % state_update_interval) == 0:
                dl.display_rover_state(connection)

            # update the FPS counter
            fps.update()
        except Exception as e:
            logging.error("Unexpected error while streaming.", exc_info=True)
            break

    logging.info("Stopping recording...")

    # stop recording
    pipeline.stop()
    time.sleep(1)
    config = None
    pipeline = None
    
    # stop the timer and display FPS information
    fps.stop()
    logging.info("Elapsed time: {:.2f}".format(fps.elapsed()))
    logging.info("Approx. FPS: {:.2f}".format(fps.fps()))
    logging.info("Indexing the recording...")
    time.sleep(10)
    logging.info("Recording complete.")


if __name__ == "__main__":
    prepare_log_file("rover_collect.log")
    print("Rover Data Collect Program.")
    print(f"Args: {sys.argv[1:]}")

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--output", type=str, help="Path to output file(s).")
    parser.add_argument("-port", "--port", type=str, help="Telemetry port to Ardupilot interface.")
    args = parser.parse_args()

    # port should look something like: "/dev/ttyUSB0"  # UART
    if args.port is not None:
        port = args.port
    else:
        port = DEFAULT_PORT

    # path should be something like: "/media/usafa/drone_data/"
    if args.output is not None:
        storage_root = args.output
    else:
        storage_root = DEFAULT_DATA_PATH

    # Create mavlink connection to device
    print(f"Connecting to autopilot on port {port} at baud rate {DEFAULT_BAUD}...")
    connection = dl.connect_device(port, DEFAULT_BAUD)
    print("Connection established.")
    device_channel_msg(connection)
    dl.display_vehicle_state(connection)

    while True:
        print("Arm vehicle to start new recording.")
        print("(CTRL-C to stop process)")
        while not connection.armed:
            time.sleep(1)

        start=time.time()
        print("Vehicle armed.")
        print("Starting recording...")
        print("(Disarm vehicle to stop recording.,)")
        bag_file = time.strftime("cloning_%Y%m%d-%H%M%S") + ".bag"
        # full path where to store data
        bag_file = storage_root + bag_file
        log_file = bag_file.replace(".bag", ".log")
        prepare_log_file(log_file)
        collect_data(bag_file)
        print("Recording stopped")
        print("Time Elapsed : " + str((time.time()-start)/60))
