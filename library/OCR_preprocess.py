from math import pi
import cv2
import numpy as np
from scipy import ndimage

EPS = pi / 4
PERCENT_OF_EXTREME_NUMBERS = 20


class OCRPreprocess:
    @staticmethod
    def preprocess_image(image):
        image = OCRPreprocess.rotate_image(image)
        image = OCRPreprocess.binarize(image)
        image = OCRPreprocess.morphology(image)
        return image

    @staticmethod
    def rotate_image(image):
        def remove_extreme_numbers(lst, x):
            sorted_lst = sorted(lst)
            to_remove = int(len(lst) * x / 100)
            trimmed_lst = sorted_lst[to_remove:-to_remove]
            return trimmed_lst

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
        thetas = []
        if lines is not None:
            for line in lines:
                _, theta = line[0]
                if -EPS < theta < EPS:
                    theta += pi / 2
                elif -EPS < theta - pi / 2 < EPS:
                    pass
                elif -EPS < theta - pi < EPS:
                    theta -= pi / 2
                else:
                    continue
                thetas.append(theta)
        thetas = remove_extreme_numbers(sorted(thetas), PERCENT_OF_EXTREME_NUMBERS)
        theta = sum(thetas) / len(thetas) if len(thetas) != 0 else pi / 2
        angle = 180 * theta / pi - 90
        img_rotated = ndimage.rotate(image, angle)
        return img_rotated

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
