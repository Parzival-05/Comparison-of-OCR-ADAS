import time
from easyocr import Reader
import pytesseract


class OCR:
    def __init__(self, gpu=0):
        self.easyOCRModel = Reader(["ru"], gpu=gpu > 0)

    def easyOCR_ocr(self, image):
        start = time.time()
        results = self.easyOCRModel.readtext(image)
        timer = time.time() - start
        text_res = []
        for _, text, _ in results:
            text_res.extend(text.lower().split())
        text_res.sort()
        return text_res, timer

    def tesseract_ocr(
        self,
        image,
    ):
        start = time.time()
        results = pytesseract.image_to_string(image, lang="rus")
        timer = time.time() - start
        text_res = sorted(map(lambda x: x.lower(), results.split()))
        return text_res, timer

    def ocrById(self, id: int, image, preprocessing_timer: int):
        if id == 0:
            ocr = self.easyOCR_ocr
        elif id == 1:
            ocr = self.tesseract_ocr
        else:
            raise Exception("Unkown OCR ID")
        text_res, timer = ocr(image)
        return text_res, timer + preprocessing_timer
