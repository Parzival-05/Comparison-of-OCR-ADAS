import cv2
import argparse

from library.stable_recognized_objects import StableRecognizedObjects
from library.road_signs_detection import RoadSignsDetection


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to input video to be OCR'd")
ap.add_argument("-g", "--gpu", type=int, default=-1, help="using GPU")
ap.add_argument(
    "-m", "--model", type=str, default="models/yolov8m.pt", help="path to model"
)
ap.add_argument(
    "-gtd", "--gtd", type=str, default="data.txt", help="path to ground truth data"
)
ap.add_argument(
    "-r",
    "--results",
    type=str,
    default="results.csv",
    help="path to csv results file (results.csv by default)",
)
args = vars(ap.parse_args())

vs = cv2.VideoCapture(args["video"])
roadSignsDetection = RoadSignsDetection(args["model"])
stableRecognizedObjects = StableRecognizedObjects(
    roadSignsDetection=roadSignsDetection, gtd=args["gtd"], results=args["results"]
)

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
