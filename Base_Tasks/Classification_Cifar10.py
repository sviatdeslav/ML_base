import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Загрузка датасета CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Нормализация пикселей в диапазон [0, 1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# Преобразование меток в one-hot encoding 
# (метод преобразования категориальных данных в числовой формат)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Выделение 5000 примеров для валидации
x_val = x_train[:5000]
y_val = y_train[:5000]
# Остальные данные для обучения
x_train = x_train[5000:]
y_train = y_train[5000:]

# Создание свёрточной модели
model = models.Sequential([
    # Блок 1
    layers.Conv2D(32, (3,3), activation='relu', padding='same',  # 32 фильтра 3x3
             kernel_regularizer=regularizers.l2(1e-4),       # L2-регуляризация
             input_shape=(32,32,3)),                         # Вход 32x32 RGB
    layers.BatchNormalization(),                                 # Нормализация активаций
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),                                  # Уменьшение размерности
    layers.Dropout(0.2),                                          # Регуляризация

    # Блок 2
    layers.Conv2D(64, (3,3), activation='relu', padding='same',
                 kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu', padding='same',
                 kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.3),

    # Блок 3
    layers.Conv2D(128, (3,3), activation='relu', padding='same',
                 kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3,3), activation='relu', padding='same',
                 kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.4),

    # Полносвязные слои
    layers.Flatten(),                                            # Преобразование в вектор
    layers.Dense(128, activation='relu',                         # Полносвязный слой
           kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),                                         # Сильный dropout
    layers.Dense(10, activation='softmax')                       # Выходной слой (10 классов)
])

# Компиляция модели
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Оптимизатор Adam
             loss='categorical_crossentropy',          # Функция потерь
             metrics=['accuracy'])                     # Метрика - точность

early_stopping = EarlyStopping(monitor='val_accuracy',  # Остановка при отсутствии прогресса
                             patience=15,             # 15 эпох ожидания
                             restore_best_weights=True)  # Возврат лучших весов

reduce_lr = ReduceLROnPlateau(monitor='val_loss',     # Уменьшение learning rate
                             factor=0.2,             # Умножение LR на 0.2
                             patience=5,             # после 5 эпох без улучшений
                             min_lr=1e-5)           # Минимальный LR

# Обучение
history = model.fit(x_train, y_train,
                   epochs=100,                      # Макс. 100 эпох
                   batch_size=64,                    # Размер батча
                   validation_data=(x_val, y_val),   # Валидационные данные
                   callbacks=[early_stopping, reduce_lr])  # Callbacks

# Оценка
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc*100:.2f}%')        # Точность на тестовых данных

# Вывод архитектуры
model.summary()  # Печать структуры модели с параметрами
