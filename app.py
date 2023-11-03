import sys

import numpy as np
from PyQt6.QtCore import QDir, Qt
from PyQt6.QtWidgets import QMainWindow, QLabel, QPushButton, QComboBox, QFileDialog, QApplication, QVBoxLayout, \
    QHBoxLayout, QWidget, QFrame, QSizePolicy, QSplitter
from PyQt6.QtGui import QPixmap, QImage
import base64
import json
import time
from collections import deque, UserList, UserDict, defaultdict
from concurrent.futures import as_completed
from threading import Thread
import cv2
import requests
from requests_futures.sessions import FuturesSession
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

global exit_flag
global batch_size
global input_source
global default_fps
global queue_threshold
global thread1, thread2, thread3


class DataStructure:
    def __init__(self, max_history=10):
        self.data = defaultdict(list)
        self.max_history = max_history

    def add(self, plate_id, license_plate):
        """
        Add a new license plate to the history of a given plate ID.

        Parameters:
            plate_id (str): The ID of the license plate.
            license_plate (str): The license plate to be added.

        Returns:
            None
        """
        string_history = self.data[plate_id]
        string_history.append(license_plate)

        if len(string_history) > self.max_history:
            string_history.pop(0)

    def get_most_frequent_license_plate(self, plate_id):
        """
        Returns the most frequent license plate for a given plate ID.

        Parameters:
            plate_id (str): The ID of the license plate.

        Returns:
            str: The most frequent license plate for the given plate ID. If no recent plates exist
                 for the given plate ID, returns "unknown".
        """
        if plate_id in self.data:
            recent_plates = self.data[plate_id][-self.max_history:]
            if recent_plates:
                plate_counts = defaultdict(int)
                for string in recent_plates:
                    plate_counts[string] += 1
                most_recent_plate = max(plate_counts, key=plate_counts.get)
                return most_recent_plate
        return "unknown"


class ByteTrackArgument:
    track_thresh = 0.5  # High_threshold
    track_buffer = 50  # Number of frame lost tracklets are kept
    match_thresh = 0.8  # Matching threshold for first stage linear assignment
    aspect_ratio_thresh = 10.0  # Minimum bounding box aspect ratio
    min_box_area = 1.0  # Minimum bounding box area
    mot20 = False  # If used, bounding boxes are not clipped.


MIN_THRESHOLD = 0.001


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.input_image = None
        self.worker_thread = []

        self.setMinimumSize(1280, 720)

        # Tạo label để hiển thị đầu vào và đặt kích thước cố định cho label
        self.label_input = QLabel()
        # self.label_input.setSizePolicy(640, 480)
        self.label_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.label_input.setFrameShape(QFrame.Shape.Box)  # Đặt kiểu viền cho label đầu vào
        self.label_input.setFrameShadow(QFrame.Shadow.Sunken)  # Đặt hiệu ứng bóng nổi cho label đầu vào
        self.label_input.setLineWidth(2)  # Đặt độ rộng của viền cho label đầu vào
        self.label_input.setMidLineWidth(1)

        # Tạo label để hiển thị đầu ra và đặt kích thước cố định cho label
        self.label_output = QLabel()
        # self.label_output.setFixedSize(640, 480)
        self.label_output.setFrameShape(QFrame.Shape.Box)  # Đặt kiểu viền cho label đầu ra
        self.label_output.setFrameShadow(QFrame.Shadow.Sunken)  # Đặt hiệu ứng bóng nổi cho label đầu ra
        self.label_output.setLineWidth(2)  # Đặt độ rộng của viền cho label đầu ra
        self.label_output.setMidLineWidth(1)

        # Tạo button để chọn loại đầu vào
        self.combo_box_input = QComboBox()
        self.combo_box_input.addItems(["Video", "Camera"])
        self.combo_box_input.currentIndexChanged.connect(self.on_combo_box_input_changed)

        # Tạo button để tải ảnh hoặc video
        self.button_load_video = QPushButton("Load video")
        self.button_load_video.clicked.connect(self.on_load_image_clicked)

        # Tạo button để phát hiện và nhận diện biển số xe
        self.button_detect = QPushButton("Bắt đầu")
        self.button_detect.setEnabled(False)
        self.button_detect.clicked.connect(self.on_start_detect)

        self.button_stop = QPushButton("Dừng")
        self.button_stop.setEnabled(False)
        self.button_stop.clicked.connect(self.on_stop_detect)

        # Thêm các widget vào layout
        input_layout = QVBoxLayout()
        input_layout.addWidget(self.label_input)
        input_layout.addWidget(self.combo_box_input)
        input_layout.addWidget(self.button_load_video)

        output_layout = QVBoxLayout()
        output_layout.addWidget(self.label_output)
        output_layout.addWidget(self.button_detect)
        output_layout.addWidget(self.button_stop)

        input_frame = QFrame()
        input_frame.setLayout(input_layout)

        output_frame = QFrame()
        output_frame.setLayout(output_layout)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(input_frame)
        self.splitter.addWidget(output_frame)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.splitter)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def on_combo_box_input_changed(self, index):
        global input_source
        selected_input = self.combo_box_input.itemText(index)

        if selected_input == "Video":
            self.button_load_video.setEnabled(True)
            input_source = None
            self.button_detect.setEnabled(False)
            pass
        elif selected_input == "Camera":
            self.button_load_video.setEnabled(False)
            self.button_detect.setEnabled(True)
            input_source = 0
            pass

    def on_load_image_clicked(self):
        global input_source
        # Mở hộp thoại để chọn ảnh hoặc video
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        # if self.combo_box_input.currentText() == "Ảnh":
        #     file_dialog.setNameFilters(["Ảnh (*.jpg *.jpeg *.png)"])

        if self.combo_box_input.currentText() == "Video":
            file_dialog.setNameFilters(["Video (*.mp4 *.avi *.mov)"])
        elif self.combo_box_input.currentText() == "Camera":
            self.cap = cv2.VideoCapture(0)
            image = self.cap.read()[1]
        if file_dialog.exec():
            # Đọc ảnh hoặc video từ file
            file_path = file_dialog.selectedFiles()[0]
            input_source = file_path
            self.button_detect.setEnabled(True)

    def on_start_detect(self):
        global thread1, thread2, thread3
        self.button_detect.setEnabled(False)
        self.button_stop.setEnabled(True)
        thread1 = Thread(target=read_frames, args=())
        thread2 = Thread(target=batch_and_send_frames, args=())
        thread3 = Thread(target=show_frames, args=())
        thread1.start()
        thread2.start()
        thread3.start()
        self.label_output.setText("Không tìm thấy biển số xe")

    def on_stop_detect(self):
        global thread1, thread2, thread3
        self.button_stop.setEnabled(False)
        global exit_flag
        exit_flag = True
        queue_frame.clear()
        queue_response.clear()
        thread1.join()
        thread2.join()
        thread3.join()
        self.button_detect.setEnabled(True)
        exit_flag = False

    def set_input_image(self, cv2_image):
        image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        # Create a QImage from the cv2 image
        height, width, channel = image.shape
        q_image = QImage(image.data, width, height, width * channel, QImage.Format.Format_RGB888)

        # Create a QPixmap from the QImage
        input_pixmap = QPixmap.fromImage(q_image)

        # Hiển thị ảnh lên label đầu vào và tự động co dãn
        input_pixmap = input_pixmap.scaledToWidth(window.label_input.width(),
                                                  mode=Qt.TransformationMode.SmoothTransformation)
        self.label_input.setPixmap(input_pixmap)

    def set_output_image(self, cv2_image):
        image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        # Create a QImage from the cv2 image
        height, width, channel = image.shape
        q_image = QImage(image.data, width, height, width * channel, QImage.Format.Format_RGB888)

        # Create a QPixmap from the QImage
        output_pixmap = QPixmap.fromImage(q_image)

        # Hiển thị ảnh lên label đầu vào và tự động co dãn
        output_pixmap = output_pixmap.scaledToWidth(window.label_input.width(),
                                                    mode=Qt.TransformationMode.SmoothTransformation)
        self.label_output.setPixmap(output_pixmap)


def read_frames():
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    global exit_flag
    global default_fps
    global queue_threshold

    # If Reading a camera, we convert to int
    try:
        device = int(input_source)
    except:
        device = input_source

    cap = cv2.VideoCapture(device)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    frame_cnt = 0
    # default_fps = cap.get(cv2.CAP_PROP_FPS)
    default_fps = 30.0
    queue_threshold = 0.5 * default_fps
    delay_time = 0
    last_frame_time = time.time()

    # Read until video is completed
    while cap.isOpened() and not exit_flag:
        # Calculate the delay time to maintain the target fps
        delay_time = (1.0 / default_fps) - (time.time() - last_frame_time)
        if delay_time > 0:
            time.sleep(delay_time)

        # Update the last frame time
        last_frame_time = time.time()

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            queue_frame.append(frame)
            frame_cnt += 1

            queue_frame_size = len(queue_frame)
            if queue_frame_size > queue_threshold:
                delay_time = (queue_frame_size - queue_threshold) / queue_threshold
                time.sleep(delay_time)

        # Break the loop
        else:
            break

    print("Done reading {} frames".format(frame_cnt))

    # When everything done, release the video capture object
    cap.release()

    # Closes after processing finished
    exit_flag = True


def send_frames(payload, snd_cnt):
    old_snd_cnt = snd_cnt

    snd_cnt += len(payload)
    response = requests.post(api, json=payload, headers=headers)

    # results = json.loads(response.text)
    # print("Old: {}, New: {} Data: {}".format(old_snd_cnt, snd_cnt, len(results)))
    # for i in range(len(results)):
    #     response_dct[old_snd_cnt + i] = results[i]
    return response, snd_cnt


def calculate_fps(start_time, snd_cnt):
    end_time = time.time()

    fps = 1.0 * batch_size / (end_time - start_time)

    # print(
    #     "With Batch Size {}, FPS at frame number {} is {:.1f}".format(
    #         batch_size, snd_cnt, fps
    #     )
    # )
    return fps


def batch_and_send_frames():
    # Initialize variables
    count, exit_cnt, snd_cnt, log_cnt = 0, 0, 0, 20
    frames = []
    payload, futures = {}, []
    start_time = time.time()
    fps = 0
    session = FuturesSession()

    while not exit_flag:
        # Batch the frames into a dict payload
        while queue_frame and count < batch_size:
            frame = queue_frame.popleft()
            frames.append(frame)

            data = cv2.imencode(".jpg", frame)[1].tobytes()
            im_b64 = base64.b64encode(data).decode("utf8")
            payload[str(count)] = im_b64
            count += 1

        if count >= batch_size:

            response, snd_cnt = send_frames(payload, snd_cnt)

            results = json.loads(response.text)
            for frame, result in zip(frames, results):
                queue_response.append([frame, result])

            if snd_cnt % log_cnt == 0:
                # Calculate FPS
                fps = calculate_fps(start_time, snd_cnt)

                # Printing the response
                # print(response.content.decode("UTF-8"))

            # Reset for next batch
            start_time = time.time()
            payload = {}
            count = 0
            frames.clear()

        # Sleep for 1 ms before trying to send next batch of frames
        time.sleep(0.001)

    # Send any remaining frames
    # _, snd_cnt = send_frames(payload, snd_cnt)
    # print(
    #     "With Batch Size {}, FPS at frame number {} is {:.1f}".format(
    #         batch_size, snd_cnt, fps
    #     )
    # )


def show_frames():
    # Initialize variables
    tracker = BYTETracker(ByteTrackArgument)
    data_struct = DataStructure()

    while True:
        if queue_response:
            frame, data = queue_response.popleft()
            height, width = frame.shape[:2]
            dets = []
            license_plate_list = []
            show_fps = (len(queue_response) / queue_threshold) * default_fps
            show_string = "FPS: {} queue_frame: {}, queue_response: {}".format(show_fps, len(queue_frame),
                                                                               len(queue_response))

            # print(show_string)

            window.set_input_image(frame)

            cv2.putText(frame, show_string, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 225), 3)

            if data["num_plate"] > 0:

                # Tracking
                for pos in range(data["num_plate"]):
                    boxes = data["boxes"][pos]
                    score = data["score"][pos]

                    det = boxes
                    det.append(score)
                    dets.append(det)
                online_targets = tracker.update(np.array(dets), [height, width], [height, width])
                print(online_targets)

                for pos in range(data["num_plate"]):
                    boxes = data["boxes"][pos]
                    license_plate = data["license_plate"][pos]

                    x1 = int(boxes[0])
                    y1 = int(boxes[1])
                    x2 = int(boxes[2])
                    y2 = int(boxes[3])

                    if len(online_targets) > pos:
                        plate_id = online_targets[pos].track_id
                        if license_plate != "unknown":
                            data_struct.add(plate_id, license_plate)
                        license_plate = "ID:{} {}".format(plate_id, data_struct.get_most_frequent_license_plate(plate_id))
                        license_plate_list.append(license_plate)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 225), thickness=2)
                    cv2.putText(frame, license_plate, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                cv2.putText(frame, ", ".join(license_plate_list), (0, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (0, 255, 0), 3)

            # cv2.imshow("Result", frame)
            window.set_output_image(frame)
            if show_fps > 0:
                time.sleep(1.0 / show_fps)
            cv2.waitKey(1)
        elif exit_flag:
            break


if __name__ == "__main__":
    # initialize
    batch_size = 4
    # input_source = 0
    # default_fps = 30
    # queue_threshold = 0.5 * default_fps
    exit_flag = False

    # Read frames are placed here and then processed
    queue_frame = deque([])
    api = "http://192.168.1.38:8080/predictions/plate"
    headers = {"Content-type": "application/json", "Accept": "text/plain"}

    queue_response = deque([])

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
