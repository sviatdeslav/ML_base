# Задача 1. Простейшие операции над тензорами
import torch
# 1. Создаем тензоры a (3x4) и b (12,)
a = torch.rand(3, 4)  # Вещественные числа от 0 до 1
b = torch.rand(12)    # Одномерный тензор из 12 элементов
# 2. Изменяем форму тензора b на (2, 2, 3)
c = b.view(2, 2, 3)   # view в PyTorch аналогичен reshape в NumPy
# Или: c = b.reshape(2, 2, 3)
# 3. Выводим первый столбец матрицы a
first_column = a[:, 0] # Срез всех строк (:) и первого столбца (0)
print("Первый столбец матрицы a:\n", first_column)


# Задача 2. Арифметические операции над тензорами
# 1. Создаем тензоры a (5x2) и b (1x10)
a = torch.rand(5, 2)
b = torch.rand(1, 10)
# 2. Изменяем форму тензора b на (5, 2)
c = b.view(5, 2)  # Важно: общее количество элементов должно совпадать
# 3. Арифметические операции
add = a + c   # Поэлементное сложение
sub = a - c   # Поэлементное вычитание
mul = a * c   # Поэлементное умножение (не матричное!)
div = a / c   # Поэлементное деление
print("Сложение:\n", add)
print("Вычитание:\n", sub)
print("Умножение:\n", mul)
print("Деление:\n", div)


# Задача 3. Функции сравнения, агрегации
import matplotlib.pyplot as plt
# 1. Создаем тензор (100, 780, 780, 3) - имитация 100 RGB-изображений
images = torch.rand(100, 780, 780, 3)  # Значения от 0 до 1
# 2. Отображаем первое изображение
plt.imshow(images[0].numpy())  # Конвертируем в NumPy для отображения
plt.title("Первое изображение")
plt.axis('off')
plt.show()
# 3. Среднее по 1-ой оси (усреднение по всем изображениям)
mean_axis0 = torch.mean(images, dim=0)  # Результат: (780, 780, 3)
plt.imshow(mean_axis0.numpy())
plt.title("Среднее по всем изображениям")
plt.axis('off')
plt.show()
# 4. Среднее по 4-ой оси (усреднение цветовых каналов)
mean_axis3 = torch.mean(images, dim=3)  # Результат: (100, 780, 780)
plt.imshow(mean_axis3[0].numpy(), cmap='gray')  # Первое ЧБ-изображение
plt.title("Усреднение каналов (первое изображение)")
plt.axis('off')
plt.show()


# Задача 4. Реализация функции прямого распространения
import torch.nn as nn
def forward_pass(X, w):
    """
    Прямой проход для одного нейрона с сигмоидной функцией активации
    X: входные данные (тензор размером [n_samples, n_features])
    w: веса (тензор размером [n_features + 1]), где w[0] - смещение (bias)
    """
    # Добавляем единичный столбец для смещения (bias term)
    X_with_bias = torch.cat([torch.ones(X.shape[0], 1), X], dim=1)
    # Вычисляем взвешенную сумму
    z = torch.matmul(X_with_bias, w)
    # Применяем сигмоидную функцию активации
    output = torch.sigmoid(z)
    return output
  
# Пример использования:
X = torch.randn(10, 3)  # 10 образцов, 3 признака
w = torch.randn(4)       # 3 веса + 1 смещение
output = forward_pass(X, w)
print("Выход нейрона:", output)


# Задача 5. Работа с CPU|GPU
from torch.autograd import Variable
# 1. Создаем тензоры на CPU
a = torch.empty(2, 3, 4).uniform_()  # Равномерное распределение [0, 1)
b = torch.empty(2, 3, 4).uniform_()
# 2. Копируем на GPU (если доступен)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a_gpu = a.to(device)
b_gpu = b.to(device)
# Вычисляем сумму и разность на GPU
sum_gpu = a_gpu + b_gpu
diff_gpu = a_gpu - b_gpu
print("Сумма на GPU:", sum_gpu)
print("Разность на GPU:", diff_gpu)
# 4. Перемещаем на CPU и оборачиваем в Variable
b_cpu = b_gpu.cpu()
a_cpu = a_gpu.cpu()
# Для PyTorch >= 0.4.0 Variable объединены с Tensor, но можно явно указать requires_grad
b_var = b_cpu.clone().requires_grad_(True)
a_var = a_cpu.clone().requires_grad_(True)
# 5. Вычисляем MSE и градиенты
L = torch.mean((b_var - a_var)**2)
L.backward()  # Вычисляем градиенты
print("Градиент dL/db:", b_var.grad)
# 6. Получаем обычный тензор из Variable (для PyTorch >= 0.4.0 это уже тензор)
b_tensor = b_var.data  # Или просто b_var, если не нужна история вычислений
print("Тензор b:", b_tensor)
