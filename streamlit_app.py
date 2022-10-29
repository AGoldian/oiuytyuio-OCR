import streamlit as st
from PIL import Image

from languages import Languages


def get_text(img):
    return ["text"], [0.6]


def main():
    st.title("Цифровой прорыв, ЮФО. Команда \"щшгненгшщ\"")
    st.subheader("Text recognition")

    lang = st.sidebar.radio("Select language", list(Languages), format_func=lambda o: o.full_name)
    file = st.file_uploader("Load your picture", type=['png', 'jpg'])
    if file is not None:
        img = Image.open(file)
        # тут подаем на нейронку
        # рисуем bbox
        # выводим картинку с bbox
        st.image(img)

#         рисуем скор


if __name__ == '__main__':
    main()
