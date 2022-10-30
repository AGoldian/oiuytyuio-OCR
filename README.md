# Распознать текст с помощью ИИ от команды щшгненгшщ

Реализовано веб-приложение для распознавания текста на фотографиях.  
Приложение в перспективе выступает помощником при поиске изображения по привычному текстовому описанию. Интерфейс позволяет пользователю загрузить фото и в ответ выдает распознанный текст.  
Решение основано на мультиязычной нейронной сети, дообученной на предоставленных данных. Ее архитектура состоит из двух этапов. На первом работает сверточная нейронная сеть, которая определяет сам текст.  
На втором работает механизм автокоррекции, который определяет вероятность ошибки в полученном тексте и корректирует его. Для увеличения количества данных проводилась аугментация.
![image](https://github.com/AGoldian/oiuytyuio-OCR/tree/main/resources/image_2022-10-30_14-24-01.png)
# Установка
```
git clone https://github.com/AGoldian/oiuytyuio-OCR.git
```

# Запуск
Для запуска необходимо иметь Microsoft Visual Studio C++ для использования библиотеки **paddleocr**
```
streamlit run streamplit_app.py
```

# Используемое решение

* В качестве решения используется дообученная на предоставленных данных (https://github.com/PaddlePaddle/PaddleOCR/)[мульти-язычная нейронная сеть], основанная на CNN и LSTM.
* Для увеличения количества данных проводится различная аугментация с последующей целью дообучения предобученной модели. 

# Уникальность:

Уникальность решения состоит в универсальности, поскольку обрабатываются фотографии из различных условий.

# Стек используемых технологий:

`Python3`, `git`, `GitHub` - инструменты разработки  
`PaddlePaddle`, `Open-MMLab`, `streamlit`, `opencv-python`, `pandas` - фреймворки нейронных сетей
`Plotly`, `Seaborn` - инструменты визуализации  

# Сравнение моделей

В качестве устойчивого решения для распознавания текста был выбран ансамбль из 2 моделей: Open-MMLab для английского и дообученный PaddlePaddle для других. Временем инференса составило 10 мс на 1660 SUPER, так как она решает задачу распознавания текста на отложенной выборке с точностью 0.45 по метрике Левенштейна.

# Проводимые исследования

(https://habr.com/ru/post/686884/)[SVTR]


# Разработчики
| Имя                  | Роль           | Контакт              |
|----------------------|----------------|----------------------|
| Оленников Вадим   | Data Scientist | https://t.me/LTDigor |
| ---                  | ---            | ---                  |
| Калинина Анастасия  | Data Scientist | https://t.me/akallnlna  |
| ---                  | ---            | --- |       
| Серов Александр       | Data Scientist | https://t.me/thegoldian   |
| ---                  | ---            |  --- |
