import torch
import torch.nn as nn
import torch.optim as optim

# Данные (X = [x1, x2], Y = [y])
X = torch.tensor([[10., 20], [1, 1], [6, 5]], dtype=torch.float32) # Обучающие данные (вход)
Y = torch.tensor([[70.], [4], [21]], dtype=torch.float32) # Обучающие данные (выход)

# Модель (линейный слой)
model = nn.Linear(2, 1, bias=False)  # 2 входа, 1 выход, без смещения

# Функция потерь и оптимизатор
criterion = nn.MSELoss() # Минимизация по среднеквадратичной ошибке (MSE)
optimizer = optim.RMSprop(model.parameters(), lr=0.01) # Выбор оптимизатора (RMSprop)

# Обучение
for epoch in range(1000):
    optimizer.zero_grad() # Обнуление производных
    outputs = model(X) # outputs — предсказания сети, model(X) — прямой проход по модели
    loss = criterion(outputs, Y) # Расчёт потерь
    loss.backward() # Автоматическое вычисление градиентов
    optimizer.step() # Обновление весов на основе вычисленных градиентов

    if epoch % 100 == 0 or epoch == 999:
        print(f'Епоха {epoch}, Потери: {loss.item()}')

# Проверка весов
print("Веса после обучения:", model.weight.data)

# Визуализация результатов
import matplotlib.pyplot as plt
import numpy as np

t1 = np.linspace(-500, 500, 1000)
t2 = np.linspace(-500, 500, 1000)
# Истинный график
u = t1 + 3 * t2

# Извлечение значений весов из тензора
x1 = model.weight.data[0][0].detach().cpu().numpy()
x2 = model.weight.data[0][1].detach().cpu().numpy()

# Предсказанный график
y = x1 * t1 + x2 * t2

import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=t1, y=t2, z=u, mode='lines', name='Истина'))
fig.add_trace(go.Scatter3d(x=t1, y=t2, z=y, mode='lines', name='Предсказание'))
fig.update_layout(scene=dict(xaxis_title='x1', yaxis_title='x2', zaxis_title='Функция'))
fig.show()
