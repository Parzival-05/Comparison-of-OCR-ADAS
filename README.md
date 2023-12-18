# Comparison-of-OCR-ADAS

This repository contains a utility for analyzing the <b>EasyOCR</b> and <b>Tesseract</b> OCR-engines on video streams with road signs with navigation information using the YOLO object recognition model. Before using OCR, images undergo preprocessing, and after that, postprocessing.

Before using this utility, you need to make a .txt file with ground truth data (gtd): each line should have text from only one object and the text from each object should be on one line.

Run it on Linux:

- `git clone https://github.com/Parzival-05/Comparison-of-OCR-ADAS`
- `pip install -r requirements.txt`
- `apt install tesseract-ocr-rus`
- `python3 main.py -v /path/to/video -m /path/to/yolo-model -gtd /path/to/gtd -r /path/to/results`

The result of the utility is a .csv file in which its recognition results (if any) are known about each object:

- Engine ID (0 for EasyOCR and 1 for Tesseract)
- Text on the object
- Recognized words
- The distance at which the text was recognized
- Recognition time
- The sum of the Levenshtein distances for each recognized word
- The number of unrecognized words
- The number of words on the object
- The number of recognized words on the object

<h2>The results of the analysis</h2>
The results showed that although EasyOCR requires large computational resources to extract text, it can provide significantly higher text recognition accuracy than Tesseract, providing good recognition results even in bad weather conditions.

\
The following video was used for the analysis: https://youtu.be/j_CAOYYs4d0.
\
Download pretrained model here: https://www.kaggle.com/datasets/notparzival/road-signs-detection-yolov8/.
