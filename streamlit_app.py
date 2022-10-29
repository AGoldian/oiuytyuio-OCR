import os

import cv2
import numpy as np
import pandas
import streamlit as st
from PIL import Image

from languages import Languages


@st.cache(suppress_st_warning=True)
def load_data():
    return Image.open('resources/img.png')


def get_text(img, lang):
    # x, y: left
    text = [[[571.0, 192.0], [929.0, 93.0], [981.0, 287.0], [623.0, 386.0]],
            [[996.0, 218.0], [1385.0, 227.0], [1382.0, 343.0], [993.0, 333.0]],
            [[962.0, 830.0], [1242.0, 979.0], [1192.0, 1074.0], [912.0, 925.0]]]

    name = ['ёть', 'ИЛи', 'o']

    prob = [0.6760122179985046, 0.6730091571807861, 0.8272780776023865]
    return text, name, prob


def write_bbox(img, bbox, color=(0, 255, 255), thickness=4):
    pts2 = np.array(bbox, np.int32)  # Массив вершин
    pts2 = pts2.reshape((- 1, 1, 2))  # Форма многомерного массива
    return cv2.polylines(img, [pts2], True, color, thickness)

def write_bboxes(img, bboxes):
    res = img.copy()
    for i in range(len(bboxes)):
        res = write_bbox(res, bboxes[i])
    return res


def main():
    st.title("Цифровой прорыв. Команда \"щшгненгшщ\"")

    img = load_data()
    file = st.file_uploader("Load your picture", type=['png', 'jpg'])
    lang = st.sidebar.radio("Select language", list(Languages), format_func=lambda o: o.full_name)

    # load default image
    img = load_data() if file is None else Image.open(file)
    img_arr = np.array(img)

    # use neuro network
    bboxes, texts, probs = get_text(img_arr, lang)

    # draw bboxes
    img_bboxes = write_bboxes(img_arr, bboxes)

    # show images
    col1, col2 = st.columns(2)
    with col1:
        st.text("Your uploaded photo:")
        st.image(img)
    with col2:
        st.text("Text in the photo:")
        st.image(img_bboxes)

    st.table(pandas.DataFrame(list(zip(texts, probs)), columns=['Text', 'Probability']))


if __name__ == '__main__':
    main()
