import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

# Загрузка датасета
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Отображение первых 30 изображений из обучающей выборки
plt.figure(figsize=(10,5))
for i in range(30):
    plt.subplot(5,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
plt.show()

model = keras.Sequential([ # Создание последовательной нейронной сети
    Flatten(input_shape=(28, 28, 1)), # Преобразование многомерных входных данные в одномерный вектор (содержит размерность изображения 28, 28 и канал 1 (чёрно-белый))
    Dense(200, activation='relu'), # Полносвязный слой
    Dense(10, activation='softmax')]) #  Полносвязный выходной слой
print(model.summary())     # Вывод структуры НС в консоль

# Нормализация от 0 до 1 (изображения 0-255)
x_train = x_train / 255
x_test = x_test / 255

# Для того, чтобы каждый из 10 выходных нейронов соответствовал номеру цифры
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

model.compile(optimizer='adam', # Оптимизатор
              loss='categorical_crossentropy', # Функция потерь
              metrics=['accuracy']) # Метрика

model.fit( # Обучение
          x_train, # Обучающие данные
          y_train_cat, # Выходные данные (ответы)
          batch_size=32, # Количество образцов, которые обрабатываются перед одним обновлением весов
          epochs=10, # Количество эпох
          validation_split=0.2) # Автоматически разделяет данные на обучение и валидацию - 20% данных используется для валидации

model.evaluate(x_test, y_test_cat) # Возвращение значений метрик по тестовым данным после обучения

# Выбор картинки для вывода результата
n = 99
x = np.expand_dims(x_test[n], axis=0) #  добавляет размерность батча(1): (1, 28, 28). Нейросети ожидают вход с размерностью (batch_size, height, width)
res = model.predict(x) # Предсказание обученной моделью
print(res) # Возвращает предсказания (значения 10 каналов выхода)

print(np.argmax(res)) # Возвращает индекс наиболее вероятного выхода

plt.imshow(x_test[n], cmap=plt.cm.binary) # Визуализация рассматриваемого числа
plt.show()
