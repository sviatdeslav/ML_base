import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
import requests
from io import StringIO

# 1. Загрузка и предварительная обработка данных
url = "http://tk.ulstu.ru/lib/info/usdrub.txt"
df = pd.read_csv(url, sep=';', header=None, 
                 names=['TICKER', 'PER', 'DATE', 'TIME', 'CLOSE'])

# 2. Преобразование CLOSE в числа и удаление заголовка (если он есть)
df['CLOSE'] = pd.to_numeric(df['CLOSE'], errors='coerce')  # Преобразуем в числа, нечисловые -> NaN
df = df.dropna()  # Удаляем строки с NaN (включая возможный заголовок)
close_prices = df['CLOSE'].values
# Визуализация исходных данных
plt.figure(figsize=(15, 5))
plt.plot(close_prices)
plt.title('История курса USD/RUB')
plt.xlabel('Время (часы)')
plt.ylabel('Курс закрытия')

# 3. Подготовка данных для нейросети
N = 50  # Количество предыдущих значений для прогноза
X, y = [], []

# Объединенная нормализация
scaler = MinMaxScaler()
close_prices_scaled = scaler.fit_transform(close_prices.reshape(-1, 1)).flatten()

# X и y берутся из уже нормализованных данных
for i in range(len(close_prices_scaled) - N):
    X.append(close_prices_scaled[i:i+N])
    y.append(close_prices_scaled[i+N])
  
X = np.array(X) # Преобразование в numpy формат
y = np.array(y)

# 4. Разбиение на обучающую, проверочную и тестовую выборки
test_size = int(0.2 * len(X))
val_size = int(0.1 * len(X))

X_train, y_train = X[:-test_size-val_size], y[:-test_size-val_size]
X_val, y_val = X[-test_size-val_size:-test_size], y[-test_size-val_size:-test_size]
X_test, y_test = X[-test_size:], y[-test_size:]

# 5. Создание и обучение модели
# Создаем последовательную модель (линейный стек слоев)
model = Sequential([
    LSTM(128, activation='relu', input_shape=(N, 1)), # LSTM слой с 128 нейронами
    # для обработки временных зависимостей, принимает на вход N временных шагов с 1 признаком
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(1) # Выходной слой с 1 нейроном (для регрессии)
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mse',
              metrics=['mae'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

history = model.fit(X_train, y_train,
                    epochs=150,
                    batch_size=64,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping, reduce_lr],
                    verbose=1)

test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"\nTest MAE: {test_mae:.4f}")  # Вывод средней абсолютной ошибки

# Прогнозирование на тестовых данных
predictions = model.predict(X_test)

# Денормализация данных:
# Возвращаем данные к исходному масштабу
y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
predictions_orig = scaler.inverse_transform(predictions).flatten()

# Визуализация результатов
plt.figure(figsize=(15, 5))
plt.plot(y_test_orig, label='Фактические значения', color='r')
plt.plot(predictions_orig, label='Прогноз', color='k')
plt.title('Прогнозирование курса USD/RUB')
plt.xlabel('Время (часы)')
plt.ylabel('Курс закрытия')
plt.legend()
plt.show()
