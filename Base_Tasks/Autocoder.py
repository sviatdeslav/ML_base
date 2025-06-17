import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# 1. Загрузка и подготовка данных
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

# Нормализация и добавление размерности канала
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# 2. Создание автокодировщика
latent_dim = 2  # Размерность скрытого пространства

# Энкодер
encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_inputs)
x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation='relu')(x)
z = layers.Dense(latent_dim, name='latent_vector')(x)

encoder = keras.Model(encoder_inputs, z, name='encoder')

# Декодер
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7*7*64, activation='relu')(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)

decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')

# Автокодировщик
autoencoder = keras.Model(encoder_inputs, decoder(encoder(encoder_inputs)), name='autoencoder')

# 3. Компиляция и обучение
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

history = autoencoder.fit(
    x_train, x_train,
    epochs=15,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# 4. Демонстрация возможностей

# Визуализация обучения
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Autoencoder Training')
plt.show()

# Пример реконструкции
n = 10  # Количество примеров
reconstructed = autoencoder.predict(x_test[:n])

plt.figure(figsize=(20, 4))
for i in range(n):
    # Оригинал
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Реконструкция
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.suptitle('Original vs Reconstructed')
plt.show()

# Генерация новых изображений
n = 15  # 15x15 grid
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

grid_x = np.linspace(-3, 3, n)
grid_y = np.linspace(-3, 3, n)

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='gray')
plt.title('Generated Digits from Latent Space')
plt.axis('off')
plt.show()
