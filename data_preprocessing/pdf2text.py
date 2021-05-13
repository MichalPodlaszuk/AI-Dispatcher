import os
import cv2
import pytesseract
from pdf2image import convert_from_path
from functools import lru_cache

@lru_cache
def pdf2text(filename):
    pages = convert_from_path(filename, 350)
    img_list = []
    txt_files = []
    i = 1
    j = 1
    for page in pages:
        image_name = "Page_" + str(i) + ".jpg"
        page.save(image_name, "JPEG")
        img_list.append(image_name)
        i += 1
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    for image in img_list:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
        dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                         cv2.CHAIN_APPROX_NONE)
        im2 = img.copy()
        file = open("../data/data_raw/text/recognized" + str(j) + ".txt", "w+")
        txt_files.append("../data/data_raw/text/recognized" + str(j) + ".txt")
        file.write("")
        file.close()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cropped = im2[y:y + h, x:x + w]
            file = open("../data/data_raw/text/recognized" + str(j) + ".txt", "r+")
            content = file.read()
            file.seek(0, 0)
            text = pytesseract.image_to_string(cropped)
            file.write(text + '\n' + content)
            file.close()
        j += 1
        os.remove(image)
    for i in range(1, len(txt_files) + 1):
        try:
            with open('../data/data_raw/text/recognized1.txt', 'a') as file_1, open(txt_files[i], 'r+') as file_2:
                content = file_2.read()
                file_1.write('\n' + content)
                file_1.close()
                file_2.close()
                os.remove(txt_files[i])
        except Exception:
            print('FUCK YOU INDEX EXCEPTION')

pdf2text('../data/data_raw/pdf/5d72f0d0f0642.file.pdf')




