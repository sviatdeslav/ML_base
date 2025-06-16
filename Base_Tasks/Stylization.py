import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Загрузка предобученной модели VGG-19 из torchvision.models
vgg = models.vgg19(pretrained=True).features  # .features - берем только часть с свёрточными слоями (без классификатора)

# Отключаем вычисление градиентов для всех параметров сети
# !Мы не будем обучать VGG, а используем её как "фиксированный экстрактор признаков"
for param in vgg.parameters():
    param.requires_grad_(False)

# torch.cuda.is_available() проверяет наличие CUDA-совместимой видеокарты
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Перено модели VGG на выбранное устройство (GPU/CPU)
vgg.to(device)

# Функция для загрузки и предварительной обработки изображения
def load_image(img_path, max_size=512):
    # Открываем изображение с помощью PIL и конвертируем в RGB
    # (на случай, если исходное изображение в grayscale или RGBA)
    image = Image.open(img_path).convert('RGB')
    # Определяем последовательность преобразований:
    transform = transforms.Compose([
        # Изменяем размер изображения, сохраняя пропорции
        # max_size - максимальный размер по длинной стороне
        transforms.Resize(max_size),
        # Конвертируем PIL Image в torch.Tensor
        # Автоматически нормализует значения пикселей в диапазон [0, 1]
        transforms.ToTensor(),
        # Нормализуем значения каналов с помощью среднего и СКО
        # Эти значения стандартны для моделей, обученных на ImageNet
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Средние значения для RGB-каналов
            std=[0.229, 0.224, 0.225]    # Стандартные отклонения
        )
    ])
    
    # Применяем преобразования и добавляем размерность batch (unsqueeze(0))
    # Получаем тензор формы [1, 3, H, W] - (batch, channels, height, width)
    return transform(image).unsqueeze(0)

content_img = load_image("Обработать.jpg").to(device)
style_img = load_image("Стиль.jpg").to(device)
# Инициализация целевого изображения
target_img = content_img.clone().requires_grad_(True)
# Выбор слоёв для стиля и содержания
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Функции потерь
def content_loss(target_features, content_features): # Вычисляет MSE (Mean Squared Error) между активациями целевого и контентного изображения
    return torch.mean((target_features - content_features)**2) # Минимизирует разницу в пространственной структуре объектов

def gram_matrix(tensor): # Вычисляет матрицу Грама - меру корреляции между картами признаков. Захватывает текстуру и стиль, игнорируя пространственное расположение
    _, c, h, w = tensor.size()  # [batch, channels, height, width]
    tensor = tensor.view(c, h * w)  # Разворачиваем пространственные измерения
    return torch.mm(tensor, tensor.t())  # Умножение матрицы на транспонированную

def style_loss(target_features, style_features): # Сравнивает текстуры через матрицы Грама. Минимизирует разницу в распределении признаков
    G_target = gram_matrix(target_features)
    G_style = gram_matrix(style_features)
    return torch.mean((G_target - G_style)**2)

def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',   # Стиль (мелкие детали)
            '5': 'conv2_1',   # Стиль
            '10': 'conv3_1',  # Стиль + содержание
            '19': 'conv4_1', # Основное содержание
            '21': 'conv4_2', # Доп. содержание (по статье Gatys)
            '28': 'conv5_1'   # Стиль (крупные детали)
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Извлекаем признаки для content, style и target изображений
content_features = get_features(content_img, vgg)
style_features = get_features(style_img, vgg)

# Веса для разных компонент потерь
style_weight = 1e6  # Увеличиваем, если стиль недостаточно выражен
content_weight = 1   # Баланс между стилем и содержанием

# Оптимизация
optimizer = optim.Adam([target_img], lr=0.01)
epochs = 500

for epoch in range(epochs):
    # Forward pass
    target_features = get_features(target_img, vgg)
    
    # Content Loss (сравниваем глубокие слои)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])) ** 2
    
    # Style Loss (сравниваем Gram матрицы по всем слоям)
    style_loss = 0
    # Считаем потери стиля на нескольких слоях (от мелких до крупных деталей)
    for layer in ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']: # Для каждого слоя:
        target_gram = gram_matrix(target_features[layer]) # Вычисляем матрицу Грама для целевого изображения
        style_gram = gram_matrix(style_features[layer]) # Вычисляем матрицу Грама для стилевого изображения
        style_loss += torch.mean((target_gram - style_gram) ** 2) # Сравниваем их через MSE
    style_loss /= len(style_layers) # Усредняем по всем слоям
    
    # Total Loss
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    # Backpropagation
    optimizer.zero_grad() # Обнуляем градиенты
    total_loss.backward() # Вычисляем градиенты (backpropagation)
    optimizer.step() # Обновляем пиксели целевого изображения
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Content Loss: {content_loss.item():.2f}, "
              f"Style Loss: {style_loss.item():.2f}")

# Денормализация и сохранение результата
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    return tensor * std + mean

# Сохранение результата
output_img = target_img.detach().cpu().squeeze()
plt.imshow(output_img.permute(1, 2, 0))
plt.show()
