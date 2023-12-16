import time
from library.OCR import OCR
from library.OCR_preprocess import OCRPreprocess
from library.road_signs_detection import RoadSignsDetection
from library.estimate_distance import *
import csv
from Levenshtein import distance as lev

RECOGNITION_THRESHOLD_OF_WORD = 0.5
RECOGNITION_THRESHOLD_OF_WORDS = 0.5


class StableRecognizedObjects:
    def __init__(self, roadSignsDetection: RoadSignsDetection, results="result.csv"):
        self.roadSignsDetection = roadSignsDetection
        self.recognized = [[], []]
        with open("./data.txt") as f:
            self.data = list(
                map(
                    lambda words: sorted(
                        list(map(lambda word: word.lower(), words[:-1].split(" ")))
                    ),
                    f.readlines()[1::],
                )
            )
        self.filename = results
        self.OCR = OCR(self.data)
        with open(self.filename, "w") as csvfile:
            self.fieldnames = [
                "ocr",
                "ground_truth",
                "prediction",
                "distance",
                "time",
                "levs_distance",
                "not_recognized",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
        return

    def is_written(self, ocr, ground_truth):
        with open(self.filename) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == str(ocr) and row[1] == str(ground_truth):
                    return True
        return False

    def are_words_match(self, word_ground_truth, word_recognized):
        lev_metric = lev(word_ground_truth, word_recognized)
        if lev_metric <= len(word_ground_truth) * RECOGNITION_THRESHOLD_OF_WORD:
            return lev_metric
        return -1

    def is_text_recognition_correct(self, recognized_words):
        for ground_truth_words in self.data:
            amount_of_recognized_words = 0
            levs_distance = 0
            recognized_words_copy = recognized_words.copy()
            for ground_truth_word in ground_truth_words:
                min_lev = float("inf")
                for recognized_word in recognized_words_copy:
                    lev = self.are_words_match(ground_truth_word, recognized_word)
                    if lev != -1 and lev < min_lev:
                        min_lev = lev
                        word = recognized_word
                else:
                    if min_lev != float("inf"):
                        amount_of_recognized_words += 1
                        levs_distance += min_lev
                        recognized_words_copy.remove(word)
            amount_of_ground_truth_words = len(ground_truth_words)
            if (
                amount_of_recognized_words
                >= RECOGNITION_THRESHOLD_OF_WORDS * amount_of_ground_truth_words
            ):  # assuming, that information on the road-sign doesn't repeat more than once
                return (
                    ground_truth_words,
                    amount_of_ground_truth_words - amount_of_recognized_words,
                    levs_distance,
                )
        return [], -1, 0

    def get_text_recognition_results_from_boxes(self, boxes: list, image, ocr_id):
        """Returns the result of recognizing boxes"""
        for road_sign in self.roadSignsDetection.get_objects(boxes, image):
            yield self.OCR.ocrById(ocr_id, road_sign)

    def handle_boxes_of_objects(self, boxes: list, image):
        for ocr_id in range(2):
            recognized_words_from_boxes = self.get_text_recognition_results_from_boxes(
                boxes, image, ocr_id
            )
            for index, (recognized_words_from_box, timer) in enumerate(
                recognized_words_from_boxes
            ):
                print(ocr_id, recognized_words_from_box)
                (
                    ground_truth_words,
                    amount_of_not_recognized_words,
                    levs_distance,
                ) = self.is_text_recognition_correct(recognized_words_from_box)
                if ground_truth_words:
                    if not self.is_written(ocr_id, ground_truth_words):
                        with open(self.filename, "a", newline="") as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                            writer.writerow(
                                {
                                    "ocr": ocr_id,
                                    "ground_truth": ground_truth_words,
                                    "prediction": recognized_words_from_box,
                                    "distance": estimate_distance(boxes[index][3]),
                                    "time": timer,
                                    "levs_distance": levs_distance,
                                    "not_recognized": amount_of_not_recognized_words,
                                }
                            )
                    else:
                        self.data.remove(ground_truth_words)
        return
