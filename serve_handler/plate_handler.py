import io
import json
import logging
import math
import os
from collections import Counter
import numpy as np
import cv2
import requests
import torch
from torchvision import transforms
from ultralytics import YOLO
import base64
from PIL import Image
from ts.torch_handler.object_detector import ObjectDetector

logger = logging.getLogger(__name__)


def deskew(src_img, change_cons, center_threshold):
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
        return rotate_image(src_img, compute_skew(change_contrast(src_img), center_threshold))
    else:
        return rotate_image(src_img, compute_skew(src_img, center_threshold))


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


try:
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
except ImportError as error:
    XLA_AVAILABLE = False


class Yolov8Handler(ObjectDetector):
    """
    Model handler for YoloV8 bounding box model
    """

    def __init__(self):
        super(Yolov8Handler, self).__init__()
        self.image_transform = transforms.ToPILImage()

    def initialize(self, context):
        # Set device type
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif XLA_AVAILABLE:
            self.device = xm.xla_device()
        else:
            self.device = torch.device("cpu")

        # Load the model
        properties = context.system_properties
        self.manifest = context.manifest
        model_dir = properties.get("model_dir")
        self.model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            self.model_pt_path = os.path.join(model_dir, serialized_file)
        self.model = self._load_torchscript_model(self.model_pt_path)
        logger.debug("Model file %s loaded successfully", self.model_pt_path)

        self.initialized = True

    def _load_torchscript_model(self, model_pt_path):
        """Loads the PyTorch model and returns the NN model object.

        Args:
            model_pt_path (str): denotes the path of the model file.

        Returns:
            (NN Model Object) : Loads the model object.
        """

        model = YOLO(model_pt_path)
        # model.to(self.device)
        return model

    def preprocess(self, data):
        images = []
        for row in data:
            imgs = row.get("data") or row.get("body")

            if isinstance(imgs, dict):
                imgs = list(imgs.values())
            else:
                imgs = [imgs]

            for image in imgs:
                if isinstance(image, str):
                    # if the image is a string of bytesarray.
                    image = base64.b64decode(image)

                # If the image is sent as bytesarray
                if isinstance(image, (bytearray, bytes)):
                    image = Image.open(io.BytesIO(image))
                    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                else:
                    # if the image is a list
                    image = torch.FloatTensor(image)

                images.append(image)

        # return torch.stack(images).to(self.device)
        return images

    def inference(self, data, *args, **kwargs):
        outputs = []
        # results = self.model(data, *args, **kwargs)
        results = self.model(data, conf=0.6)
        # print(results)

        for result in results:
            plates = result.boxes.data.cpu().tolist()
            img = result.orig_img

            license_plate_lst = []

            for plate in plates:
                x1 = int(plate[0])
                y1 = int(plate[1])
                x2 = int(plate[2])
                y2 = int(plate[3])

                plate_img = img[y1:y2, x1:x2]

                # date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

                payload = {}
                cnt = 0
                for cc in range(0, 2):
                    for ct in range(0, 2):
                        proceessed_img = deskew(plate_img, cc, ct)

                        _, buffer = cv2.imencode('.jpg', proceessed_img)
                        encoded_image = base64.b64encode(buffer).decode('utf-8')

                        cnt += 1
                        payload[str(cnt)] = encoded_image

                        # file_path = "/home/tan/PycharmProjects/serve/examples/object_detector/yolo/yolov8/test/{}_{}_output_image_{}.jpg".format(
                        #     cc, ct, date_time)
                        # cv2.imwrite(file_path, proceessed_img)
                        # Convert the image to Base64

                api = 'http://127.0.0.1:8080/predictions/character'
                headers = {"Content-type": "application/json", "Accept": "text/plain"}

                # Send a POST request with the file
                response = requests.post(api, headers=headers, json=payload)

                license_plate = "unknown"

                if response.status_code == 200:
                    plate_lst = [item for item in json.loads(response.text) if item != "unknown"]
                    if plate_lst:
                        frequency_counter = Counter(plate_lst)
                        license_plate = frequency_counter.most_common(1)[0][0]

                plate.append(license_plate)
                license_plate_lst.append(plate)

            outputs.append(license_plate_lst)
            # outputs.append(plates)
        return [outputs]

    def postprocess(self, res):
        output = []
        for data in res[0]:
            output_data = dict()
            output_data["num_plate"] = len(data)
            output_data["boxes"] = []
            output_data["score"] = []
            output_data["classes"] = []
            output_data["license_plate"] = []

            for plate_data in data:
                output_data["boxes"].append(plate_data[:4])
                output_data["score"].append(plate_data[4])
                output_data["classes"].append(self.model.names[int(plate_data[5])])
                output_data["license_plate"].append(plate_data[6])
            # print(output_data)
            output.append(output_data)
        return [output]
