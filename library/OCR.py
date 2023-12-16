import time
from easyocr import Reader
import pytesseract
from library.OCR_preprocess import OCRPreprocess
from library.OCR_postprocess import OCRPostprocess


class OCR:
    def __init__(self, gtd, gpu=0):
        self.easyOCRModel = Reader(["ru"], gpu=gpu > 0)
        self.OCR_postprocess = OCRPostprocess(gtd)

    def easyOCR_ocr(self, image):
        results = self.easyOCRModel.readtext(image)
        text_res = []
        for _, text, _ in results:
            text_res.extend(text.lower().split())
        return text_res

    def tesseract_ocr(
        self,
        image,
    ):
        results = pytesseract.image_to_string(image, lang="rus")
        text_res = list(map(lambda x: x.lower(), results.split()))
        return text_res

    def ocrById(self, id: int, image):
        start = time.time()
        image = OCRPreprocess.preprocess_image(image)
        if id == 0:
            result = self.easyOCR_ocr(image)
        elif id == 1:
            result = self.tesseract_ocr(image)
        else:
            raise Exception("Unkown OCR ID")
        postprocessed_result = self.OCR_postprocess.postprocess(result)
        return postprocessed_result, time.time() - start
