import tensorflow as tf
from tensorflow import keras
from kerastuner import RandomSearch, HyperParameters

# 1. Загрузка данных
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 2. Определение модели для тюнинга
def build_model(hp):
    model = keras.Sequential()
    
    # Оптимизация архитектуры
    hp_units1 = hp.Int('units1', min_value=32, max_value=512, step=32)
    hp_units2 = hp.Int('units2', min_value=32, max_value=256, step=32)
    
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(units=hp_units1, activation='relu'))
    
    # Оптимизация dropout
    hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
    model.add(keras.layers.Dropout(hp_dropout))
    
    model.add(keras.layers.Dense(units=hp_units2, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    # Оптимизация learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 3. Настройка тюнера
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,  # Количество вариантов для тестирования
    executions_per_trial=2,  # Повторы для каждого варианта
    directory='mnist_tuning',
    project_name='mnist_keras_tuner'
)

# 4. Поиск лучших гиперпараметров
tuner.search(
    x_train, 
    y_train,
    epochs=5,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)]
)

# 5. Получение результатов
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
Оптимальные гиперпараметры:
- Количество нейронов 1 слоя: {best_hps.get('units1')}
- Количество нейронов 2 слоя: {best_hps.get('units2')}
- Dropout rate: {best_hps.get('dropout')}
- Learning rate: {best_hps.get('learning_rate')}
""")

# 6. Обучение лучшей модели
best_model = tuner.get_best_models(num_models=1)[0]
history = best_model.fit(
    x_train, 
    y_train, 
    epochs=10, 
    validation_split=0.2
)

# 7. Оценка на тестовых данных
test_loss, test_acc = best_model.evaluate(x_test, y_test)
print(f'\nTest accuracy: {test_acc:.4f}')

# 8. Визуализация процесса
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
