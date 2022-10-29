# Распознать текст с помощью ИИ от команды щшгненгшщ

Реализовано веб-приложение для распознавания текста на изображениях, которое может выступать в качестве инструмента поиска информации на изображении по привычному текстовому описанию.

# Установка
- `git clone https://github.com/AGoldian/oiuytyuio-OCR.git`

# Запуск
```
streamlit run streamplit_app.py
```

# Используемое решение

* В качестве решения используется дообученная на предоставленных данных (мульти-язычная нейронная сеть)[https://github.com/PaddlePaddle/PaddleOCR/], основанная на CNN и LSTM.
* Для увеличения количества данных проводится различная аугментация с последующей целью дообучения предобученной модели. 

# Уникальность:

Разработанный пайплайн является уникальным решением на рынке за счет высокоточной возможности распознавания русскоязычного текста.

# Стек используемых технологий:

`Python3`, `git`, `GitHub` - инструменты разработки  
`PaddlePaddle`, `Open-MMLab`, `streamlit`, `opencv-python`, `pandas` - фреймворки нейронных сетей
`Plotly`, `Seaborn` - инструменты визуализации  

# Сравнение моделей

В качестве устойчивого решения для распознавания текста был выбран ансамбль из 2 моделей: Open-MMLab для английского и дообученный PaddlePaddle для других. Временем инференса составило 10 мс на 1660 SUPER, так как она решает задачу распознавания текста на отложенной выборке с точностью 0.45 по метрике Левенштейна.

# Проводимые исследования

(SVTR)[https://habr.com/ru/post/686884/]


# Разработчики
| Имя                  | Роль           | Контакт              |
|----------------------|----------------|----------------------|
| Оленников Вадим   | Data Scientist | https://t.me/LTDigor |
| ---                  | ---            | ---                  |
| Калинина Анастасия  | Data Scientist | -                    |
| ---                  | ---            | https://t.me/        |                    |
| Серов Александр       | Data Scientist | -                    |
| ---                  | ---            | ---                  |