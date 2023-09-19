import argparse
import math
import os
from collections import defaultdict
from datetime import datetime

import cv2
import filetype
import numpy as np
from ultralytics import YOLO


# class DataStructure:
#     def __init__(self):
#         self.data = defaultdict(lambda: defaultdict(int))
# 
#     def add(self, plate_id, license_plate):
#         self.data[plate_id][license_plate] += 1
# 
#     def get_most_frequent_license_plate(self, plate_id):
#         if plate_id in self.data:
#             string_counts = self.data[plate_id]
#             most_frequent_plate_id = max(string_counts, key=string_counts.get)
#             return most_frequent_plate_id
#         else:
#             return "unknown"


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


class Detector:
    def __init__(self, model_detection_path="models/plate.pt", model_recognition_path="models/character.pt",
                 conf_threshold=None,
                 iou_threshold=None, device="0"):
        self.model_detection = YOLO(model_detection_path)
        self.model_recognition = YOLO(model_recognition_path)
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.names = self.model_recognition.names
        self.device = device

    def predict_image(self, img_path, show=False, save=False, addvanced_mode=False):
        """
        Predicts the content of an image.

        Args:
            img_path (str): The path of the image to be predicted.
            show (bool, optional): Whether to display the image with the predicted results. Defaults to False.
            save (bool, optional): Whether to save the image with the predicted results. Defaults to False.
            addvanced_mode (bool, optional): Whether to use advanced mode for license plate recognition. Defaults to False.

        Returns:
            None

        Raises:
            None
        """
        img = cv2.imread(img_path)
        results = self.model_detection(img, iou=self.iou_threshold, conf=self.conf_threshold, device=self.device)[
            0].boxes.cpu()
        plates = self.detection_plates(results)
        for plate_id, plate_pos in plates:
            x1 = int(plate_pos[0])
            y1 = int(plate_pos[1])
            x2 = int(plate_pos[2])
            y2 = int(plate_pos[3])

            plate_img = img[y1:y2, x1:x2]
            license_plate = "unknown"

            if addvanced_mode:
                flag = 0
                for cc in range(0, 2):
                    for ct in range(0, 2):
                        license_plate = self.recognition_plate(self.deskew(plate_img, cc, ct))
                        if license_plate != "unknown":
                            flag = 1
                            break
                    if flag == 1:
                        break
            else:
                license_plate = self.recognition_plate(plate_img)

            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 225), thickness=2)
            cv2.putText(img, license_plate, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        if show:
            cv2.imshow("Result", img)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

        if save:
            os.makedirs("outputs", exist_ok=True)
            date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            cv2.imwrite("outputs/output_image_{}.jpg".format(date_time), img)

    def predict_video(self, source, track=True, show=False, save=False, addvanced_mode=False):
        """
        Predicts video frames and performs object detection on each frame.

        Parameters:
            source (str): The path or URL of the video file to be processed.
            track (bool, optional): Whether to enable object tracking. Defaults to True.
            show (bool, optional): Whether to display the video frames. Defaults to False.
            save (bool, optional): Whether to save the processed video. Defaults to False.
            addvanced_mode (bool, optional): Whether to use advanced mode for license plate recognition. Defaults to False.

        Returns:
            None

        Raises:
            None

        """
        cap = cv2.VideoCapture(source)
        data = DataStructure()

        if save:
            os.makedirs("outputs", exist_ok=True)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            video_output_path = "outputs/output_video_{}.mp4".format(date_time)
            video_output = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30,
                                           (frame_width, frame_height))

        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                if track:
                    results = \
                        self.model_detection.track(frame, conf=self.conf_threshold, iou=self.iou_threshold,
                                                   persist=True,
                                                   tracker="cfg/tracker/botsort.yaml", device=self.device)[
                            0].boxes.cpu()
                else:
                    results = \
                        self.model_detection(frame, conf=self.conf_threshold, iou=self.iou_threshold,
                                             device=self.device)[
                            0].boxes.cpu()

                plates = self.detection_plates(results)
                for plate_id, plate_pos in plates:
                    x1 = int(plate_pos[0])
                    y1 = int(plate_pos[1])
                    x2 = int(plate_pos[2])
                    y2 = int(plate_pos[3])
                    plate_img = frame[y1:y2, x1:x2]

                    license_plate = "unknown"

                    if addvanced_mode:
                        flag = 0
                        for cc in range(0, 2):
                            for ct in range(0, 2):
                                license_plate = self.recognition_plate(self.deskew(plate_img, cc, ct))
                                if license_plate != "unknown":
                                    flag = 1
                                    break
                            if flag == 1:
                                break
                    else:
                        license_plate = self.recognition_plate(plate_img)

                    if plate_id != -1 and license_plate != "unknown":
                        data.add(plate_id, license_plate)

                    license_plate = data.get_most_frequent_license_plate(plate_id)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 225), thickness=2)
                    cv2.putText(frame, "ID: {} - {}".format(plate_id, license_plate), (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                if show:
                    cv2.imshow("Result", frame)

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if save:
                    video_output.write(frame)
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        if save:
            video_output.release()
        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def detection_plates(results):
        """
        Generates a list of tuples containing the IDs and plates detected in the given results.

        Parameters:
            results (Object): The results object containing the detected plates.

        Returns:
            List[Tuple[int, List[float]]]: A list of tuples where each tuple contains the ID and the
            coordinates of a detected plate.
        """
        plates = results.xyxy.tolist()
        ids = [-1] * len(plates)

        if results.is_track:
            ids = list(map(int, results.id.tolist()))

        return list(zip(ids, plates))

    def recognition_plate(self, plate_img):
        """
        This function takes in an image of a license plate and performs license plate recognition.

        Parameters:
            plate_img (image): The image of the license plate to be recognized.

        Returns:
            license_plate (str): The recognized license plate number as a string.

        Raises:
            unknown: If no license plate is detected or the number of bounding boxes is less than 7 or greater than 10.

        """
        plate_type = "1"
        results = self.model_recognition(plate_img, conf=self.conf_threshold, iou=self.iou_threshold)
        bb_list = results[0].boxes.data.cpu().tolist()
        if len(bb_list) == 0 or len(bb_list) < 7 or len(bb_list) > 10:
            return "unknown"
        center_list = []
        y_mean = 0
        y_sum = 0
        for bb in bb_list:
            x_c = (bb[0] + bb[2]) / 2
            y_c = (bb[1] + bb[3]) / 2
            y_sum += y_c
            center_list.append([x_c, y_c, self.names[int(bb[-1])]])

        # find 2 point to draw line
        l_point = center_list[0]
        r_point = center_list[0]
        for cp in center_list:
            if cp[0] < l_point[0]:
                l_point = cp
            if cp[0] > r_point[0]:
                r_point = cp
        for ct in center_list:
            if l_point[0] != r_point[0]:
                if not self.check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]):
                    plate_type = "2"

        y_mean = int(int(y_sum) / len(bb_list))

        # 1 line plates and 2 line plates
        line_1 = []
        line_2 = []
        license_plate = ""
        if plate_type == "2":
            for c in center_list:
                if int(c[1]) > y_mean:
                    line_2.append(c)
                else:
                    line_1.append(c)
            for l1 in sorted(line_1, key=lambda x: x[0]):
                license_plate += str(l1[2])
            license_plate += "-"
            for l2 in sorted(line_2, key=lambda x: x[0]):
                license_plate += str(l2[2])
        else:
            for l in sorted(center_list, key=lambda x: x[0]):
                license_plate += str(l[2])
        return license_plate

    def deskew(self, src_img, change_cons, center_threshold):
        """
        Generate the function comment for the given function body in a markdown code block with the correct language syntax.

        Parameters:
            src_img (Image): The source image to deskew.
            change_cons (int): A flag indicating whether to change the contrast of the image before deskewing.
            center_threshold (float): The threshold for determining the center of the image.

        Returns:
            Image: The deskewed image.
        """
        if change_cons == 1:
            return self.rotate_image(src_img, self.compute_skew(self.change_contrast(src_img), center_threshold))
        else:
            return self.rotate_image(src_img, self.compute_skew(src_img, center_threshold))

    @staticmethod
    def compute_skew(src_img, center_threshold):
        """
        Computes the skew of the source image.

        Parameters:
        - src_img: The source image to compute the skew from.
        - center_threshold: The threshold for filtering center points.

        Returns:
        - The computed skew angle in degrees.

        Note:
        - This function uses OpenCV operations to process the image.
        """
        if len(src_img.shape) == 3:
            h, w, _ = src_img.shape
        elif len(src_img.shape) == 2:
            h, w = src_img.shape
        else:
            print('upsupported image type')
        img = cv2.medianBlur(src_img, 3)
        edges = cv2.Canny(img, threshold1=30, threshold2=100, apertureSize=3, L2gradient=True)
        lines = cv2.HoughLinesP(edges, 1, math.pi / 180, 30, minLineLength=w / 1.5, maxLineGap=h / 3.0)
        if lines is None:
            return 1

        min_line = 100
        min_line_pos = 0
        for i in range(len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                center_point = [((x1 + x2) / 2), ((y1 + y2) / 2)]
                if center_threshold == 1:
                    if center_point[1] < 7:
                        continue
                if center_point[1] < min_line:
                    min_line = center_point[1]
                    min_line_pos = i

        angle = 0.0
        nlines = lines.size
        cnt = 0
        for x1, y1, x2, y2 in lines[min_line_pos]:
            ang = np.arctan2(y2 - y1, x2 - x1)
            if math.fabs(ang) <= 30:  # excluding extreme rotations
                angle += ang
                cnt += 1
        if cnt == 0:
            return 0.0
        return (angle / cnt) * 180 / math.pi

    @staticmethod
    def change_contrast(img):
        """
        Apply contrast enhancement to an image.

        Parameters:
            img (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The enhanced image.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced_img

    @staticmethod
    def rotate_image(image, angle):
        """
        Rotates an image by a given angle.

        Parameters:
            image (numpy.ndarray): The input image.
            angle (float): The angle of rotation in degrees.

        Returns:
            numpy.ndarray: The rotated image.

        """
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    @staticmethod
    def linear_equation(x1, y1, x2, y2):
        """
        Calculate the coefficients of a linear equation given two points.

        Args:
            x1 (float): The x-coordinate of the first point.
            y1 (float): The y-coordinate of the first point.
            x2 (float): The x-coordinate of the second point.
            y2 (float): The y-coordinate of the second point.

        Returns:
            Tuple[float, float]: A tuple containing the coefficients 'a' and 'b' of the linear equation, where 'a' is the slope and 'b' is the y-intercept.
        """
        b = y1 - (y2 - y1) * x1 / (x2 - x1)
        a = (y1 - b) / x1
        return a, b

    def check_point_linear(self, x, y, x1, y1, x2, y2):
        """
        Check if a given point lies on a line defined by two other points.

        Parameters:
            x (float): The x-coordinate of the point to be checked.
            y (float): The y-coordinate of the point to be checked.
            x1 (float): The x-coordinate of the first point defining the line.
            y1 (float): The y-coordinate of the first point defining the line.
            x2 (float): The x-coordinate of the second point defining the line.
            y2 (float): The y-coordinate of the second point defining the line.

        Returns:
            bool: True if the point lies on the line, False otherwise.
        """
        a, b = self.linear_equation(x1, y1, x2, y2)
        y_pred = a * x + b
        return math.isclose(y_pred, y, abs_tol=3)


def parse_options():
    parser = argparse.ArgumentParser(description='YOLOv8 Inference')
    parser.add_argument('--source', default='test_video/test.mov', help='source image or video')
    parser.add_argument('--model_detection', default='models/plate.pt', help='YOLOv8 model path')
    parser.add_argument('--model_recognition', default='models/character.pt', help='YOLOv8 model path')
    parser.add_argument('--conf', default=0.6, type=float, help='confidence threshold')
    parser.add_argument('--iou', default=0.7, type=float, help='iou threshold')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    options = parser.parse_args()
    return options


if __name__ == '__main__':
    options = parse_options()

    model = Detector(options.model_detection, options.model_recognition, options.conf, options.iou, options.device)

    if filetype.is_image(options.source):
        model.predict_image(options.source, show=True, save=True, addvanced_mode=True)
    elif filetype.is_video(options.source):
        model.predict_video(options.source, show=True, save=True, addvanced_mode=True)
    else:
        print("Unsupported file type")
