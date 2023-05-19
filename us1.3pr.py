import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('mnist.csv')
dataset = pd.DataFrame.dropna(dataset)
dataset.head()

# выделяем целевую переменную и остальную часть
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, :1].values

# Разбиваем выборку на тестовую и обучаемую
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Обчучаем модель по алгоритму деревбев принятия решений
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Выводим ошибки и различные метрики качества классификации
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
