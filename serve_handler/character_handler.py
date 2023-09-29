import io
import logging
import math
import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from ultralytics import YOLO
import base64
from PIL import Image
from ts.torch_handler.object_detector import ObjectDetector

logger = logging.getLogger(__name__)


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


def check_point_linear(x, y, x1, y1, x2, y2):
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
    a, b = linear_equation(x1, y1, x2, y2)
    y_pred = a * x + b
    return math.isclose(y_pred, y, abs_tol=3)


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
        results = self.model(data, *args, **kwargs)

        for result in results:
            license_plate = self.recognition_plate(result)
            outputs.append(license_plate)

        return [outputs]

    def postprocess(self, res):
        output = []
        for data in res:
            output.append(data)

        return output

    def recognition_plate(self, data):
        plate_type = "1"
        bb_list = data.boxes.data.cpu().tolist()
        if len(bb_list) == 0 or len(bb_list) < 7 or len(bb_list) > 10:
            return "unknown"
        center_list = []
        y_mean = 0
        y_sum = 0
        for bb in bb_list:
            x_c = (bb[0] + bb[2]) / 2
            y_c = (bb[1] + bb[3]) / 2
            y_sum += y_c
            center_list.append([x_c, y_c, self.model.names[int(bb[-1])]])

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
                if not check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]):
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
