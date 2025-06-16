# 1. Импорт БД. Вывод верхних строк
import pandas as pd
import numpy as np
import seaborn as sb

iris = sb.load_dataset("iris")  # Загружаем данные из объекта `iris.data` в формат DataFrame
iris.head()

# 2. Визуализация диаграмм рассеивания
sb.pairplot(iris, hue='species', markers=["o", "s", "D"]) # Диаграммы рассевания

# 3. Обучение модели через K_means
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
# Загрузка данных (предполагается, что df уже определен)
X_train = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_train = iris['species']
# Инициализация модели KNN
knn = KNeighborsClassifier(n_neighbors=3)
# Обучение модели
knn.fit(X_train, y_train)
# Предсказание для нового объекта
X_test = np.array([[1.2, 1.0, 2.8, 1.2]])
target = knn.predict(X_test)
print("Предсказанный класс:", target)

# 4. Вычисление точности модели
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Разделение данных на обучающую и тестовую выборки
X_train, X_holdout, y_train, y_holdout = train_test_split(
    iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']],
    iris['species'],
    test_size=0.3,
    random_state=17
)
# Обучение модели
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
# Предсказание и оценка точности
knn_pred = knn.predict(X_holdout)
accuracy = accuracy_score(y_holdout, knn_pred)
print("Точность модели:", accuracy)

# 5. Вычисление оптимального числа соседей К
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
# Определение диапазона значений K
k_list = list(range(1, 50))
cv_scores = []
# Кросс-валидация для каждого K
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, iris.iloc[:, 0:4], iris['species'], cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
# Расчет ошибки классификации
MSE = [1 - x for x in cv_scores]
# Построение графика
plt.plot(k_list, MSE)
plt.xlabel('Количество соседей (K)')
plt.ylabel('Ошибка классификации (MSE)')
plt.show()
# Поиск оптимального K
min_error = min(MSE)
optimal_k = [k for k, error in zip(k_list, MSE) if error == min_error]
print("Оптимальные значения K:", optimal_k)
