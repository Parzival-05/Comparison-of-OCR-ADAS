from ultralytics import YOLO


class RoadSignsDetection:
    def __init__(self, model="models/yolov8m.pt"):
        self.model = YOLO(model)

    def predict(self, source, **kwargs):
        return self.model(source, **kwargs)

    def get_boxes(self, result):
        """Returns boxes represented as xyxy"""
        return result.boxes.xyxy.numpy().astype(int)

    def get_objects(self, boxes, image):
        """Returns images by boxes"""
        for box in boxes:
            yield image[box[1] : box[3], box[0] : box[2]]
