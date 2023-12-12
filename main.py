import cv2
import argparse

from library.stable_recognized_objects import StableRecognizedObjects
from library.road_signs_detection import RoadSignsDetection


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to input image to be OCR'd")
ap.add_argument("-g", "--gpu", type=int, default=-1, help="using GPU")
args = vars(ap.parse_args())

vs = cv2.VideoCapture(args["video"])
roadSignsDetection = RoadSignsDetection("models/yolov8m.pt")
stableRecognizedObjects = StableRecognizedObjects(roadSignsDetection=roadSignsDetection)

frame_number = -1
while True:
    image = vs.read()[1]
    if image is None:
        break
    else:
        frame_number += 1
    image = image[0:900, 0:1920]
    results = roadSignsDetection.predict(
        image,
    )[0]
    boxes = roadSignsDetection.get_boxes(results)
    stableRecognizedObjects.handle_boxes_of_objects(boxes, image)
    print(f"Frame number: {frame_number}\n")
