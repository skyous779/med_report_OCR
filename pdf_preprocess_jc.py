#!/usr/bin/env python
# coding=utf-8
"""
Author       : Bowen Zheng
Date         : 2024-03-04 09:26:16
LastEditors  : ibowennn shihun44@163.com
LastEditTime : 2024-03-04 10:23:17
Description  : 

"""

import glob
import os
import re
import shutil

import cv2
import fitz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from img2table.document import Image
from img2table.ocr import PaddleOCR

"""
extract psa and fpsa from the report
"""


def line_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=600, minLineLength=300, maxLineGap=100
    )

    horizontal_lines = []
    vertical_lines = []
    split_line = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) > abs(y2 - y1):  # 水平线的斜率接近于0
            horizontal_lines.append(line)
            horizontal_lines = sorted(
                horizontal_lines, key=lambda x: min(x[0][1], x[0][3])
            )
        else:  # 垂直线的斜率很大，或者接近无穷大
            vertical_lines.append(line)
            vertical_lines = sorted(vertical_lines, key=lambda x: min(x[0][0], x[0][2]))
        if line[0][0] == 0:
            split_line = line

    return horizontal_lines, vertical_lines, split_line


def split_image_ud(image, split_line):
    x1, y1, x2, y2 = split_line[0]
    top_half = image[:y1, :]
    bottom_half = image[y1:, :]
    return top_half, bottom_half


def split_image_lr(image, vertical_lines):
    x1, _, _, _ = vertical_lines[0][0]
    left_half = image[:, :x1]
    right_half = image[:, x1:]
    return left_half, right_half


def extract_table(file, image, ocr):
    _, encoded_image = cv2.imencode(".png", image)
    image_bytes = encoded_image.tobytes()
    image_to_extract = Image(image_bytes)
    try:
        extract_tables = image_to_extract.extract_tables(
            ocr=ocr, min_confidence=50, borderless_tables=True
        )

        df = extract_tables[0].df

        return df
    except:
        print(f"No table found:{file}")


def extract_3lines_table(image, horizontal_lines):
    header1 = image[horizontal_lines[0][0][1] : horizontal_lines[1][0][1], :]
    table1 = image[horizontal_lines[2][0][1] : horizontal_lines[3][0][1], :]

    header2 = image[horizontal_lines[5][0][1] : horizontal_lines[6][0][1], :]
    table2 = image[horizontal_lines[7][0][1] : horizontal_lines[8][0][1], :]
    return table1, table2, header1, header2


if __name__ == "__main__":

    root_dir = "/home/data2/public/gzyy/images"
    jc_png = glob.glob(os.path.join(root_dir, "*/*jc*.png"))

    ocr = PaddleOCR(
        lang="ch"
    )  # need to run only once to download and load model into memory

    for file in tqdm(jc_png[0:10]):
        df_all = pd.DataFrame()
        img = cv2.imread(file)
        horizontal_lines, vertical_lines, split_line = line_detection(img)
        if split_line is not None:
            # table1, table2 = split_image_ud(img, split_line)
            table1, table2, header1, header2 = extract_3lines_table(
                img, horizontal_lines
            )

            if len(vertical_lines) == 1:
                if vertical_lines[0][0][1] < split_line[0][1]:
                    table1_1, table1_2 = split_image_lr(table1, vertical_lines)
                    table_to_extract = [table1_1, table1_2, table2]
                else:
                    table2_1, table2_2 = split_image_lr(table2, vertical_lines)
                    table_to_extract = [table1, table2_1, table2_2]
            elif len(vertical_lines) == 2:
                table1_1, table1_2 = split_image_lr(table1, vertical_lines)
                table2_1, table2_2 = split_image_lr(table2, vertical_lines)
                table_to_extract = [table1_1, table1_2, table2_1, table2_2]
            elif len(vertical_lines) == 0:
                table_to_extract = [table1, table2]
        else:
            header1 = img[horizontal_lines[0][0][1] : horizontal_lines[1][0][1], :]
            table1 = img[horizontal_lines[2][0][1] : horizontal_lines[3][0][1], :]
            if vertical_lines:
                table1_1, table1_2 = split_image_lr(table1, vertical_lines)
                table_to_extract = [table1_1, table1_2]
            else:
                table_to_extract = [table1]
        counter = 0
        for table in table_to_extract:
            counter += 1
            df = extract_table(file, table, ocr)
            df_all = pd.concat([df_all, df], axis=0)
            df_all.to_csv(f"../data/{counter}.csv", index=False)
