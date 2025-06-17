import numpy as np
import matplotlib.pyplot as plt

# Параметры
N = 100  # Количество точек на один класс
D = 2    # Размерность пространства
K = 9    # Количество классов

# Инициализация массивов
X = np.zeros((N * K, D))  # Матрица данных
y = np.zeros(N * K, dtype='uint8')  # Вектор меток классов

# Генерация данных
for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1.0, N)  # Радиусы от 0 до 1
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # Углы с шумом
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]  # Преобразование в декартовы координаты
    y[ix] = j  # Присвоение меток классов

# Визуализация данных
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.rainbow)  # Различные цвета для разных классов
plt.title('Игрушка дьявола', fontsize=15)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$y$', fontsize=14)
plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Преобразование для многоклассовой классификации
class DevilNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DevilNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)  # Выходной слой = числу классов
        )
    
    def forward(self, x):
        return self.layers(x)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Нормализация
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Конвертация в тензоры
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)  # LongTensor для CrossEntropyLoss
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Инициализация модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DevilNet(X_train.shape[1], num_classes=K).to(device)
criterion = nn.CrossEntropyLoss()  # Для многоклассовой классификации
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Обучение
train_losses = []
test_losses = []
epochs = 200

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train_tensor.to(device))
    loss = criterion(outputs, y_train_tensor.to(device))
    loss.backward()
    optimizer.step()
    
    # Валидация
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor.to(device))
        test_loss = criterion(test_outputs, y_test_tensor.to(device))
    
    train_losses.append(loss.item())
    test_losses.append(test_loss.item())
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# Визуализация
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.show()

# Оценка
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor.to(device))
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor.to(device)).float().mean()
    print(f'Test Accuracy: {accuracy.item():.4f}')

# Визуализация решений
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    model.eval()
    with torch.no_grad():
        Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device))
        _, Z = torch.max(Z, 1)
        Z = Z.cpu().numpy().reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.rainbow)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.rainbow, edgecolors='k')
    plt.title("Границы решений")
    plt.show()

plot_decision_boundary(model, X_test, y_test)
