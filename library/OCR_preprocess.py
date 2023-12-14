from math import pi
import cv2
import numpy as np
from scipy import ndimage

EPS = pi / 4
ACCEPTABLE_DEGREE_OF_ROTATION = 0.05  # radians


class OCRPreprocess:
    @staticmethod
    def preprocess_image(image):
        image = OCRPreprocess.rotate_image(image)
        image = OCRPreprocess.binarize(image)
        image = OCRPreprocess.morphology(image)
        return image

    @staticmethod
    def rotate_image(image):
        def remove_outliers(data):
            if len(data) == 0:
                return []
            mean = np.mean(data)
            std = np.std(data)
            lower_bound = mean - 2 * std
            upper_bound = mean + 2 * std
            return [x for x in data if x >= lower_bound and x <= upper_bound]

        def auto_canny(image, sigma=0.33):
            # compute the median of the single channel pixel intensities
            v = np.median(image)
            # apply automatic Canny edge detection using the computed median
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            edged = cv2.Canny(image, lower, upper)
            # return the edged image
            return edged

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = auto_canny(gray)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100, np.array([]))
        if lines is None:
            return image
        thetas = []
        for line in lines:
            _, theta = line[0]
            if 0 < pi / 2 - theta < EPS:
                theta = theta - pi / 2
            elif 0 < theta - pi / 2 < EPS:
                theta = pi / 2 - theta
            elif pi - theta < EPS:
                theta = theta - pi
            thetas.append(theta)
        thetas = remove_outliers(thetas)
        theta_mean = np.mean(thetas) if len(thetas) else 0
        if abs(theta_mean) > ACCEPTABLE_DEGREE_OF_ROTATION:
            return image
        return ndimage.rotate(image, 180 * theta_mean / pi)

    @staticmethod
    def binarize(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary_image = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        binary_image = cv2.bitwise_not(binary_image)
        return binary_image

    @staticmethod
    def morphology(image):
        kernel = np.zeros((3, 3), np.uint8)
        closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return closed_image
