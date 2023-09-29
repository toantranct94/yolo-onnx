import math

import cv2


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
