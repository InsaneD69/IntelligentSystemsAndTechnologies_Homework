import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# считаем и выводим Mean Absolute Error и Mean Squared Error
def get_print_mae_mse(y_test, y_pred):
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    print('Mean Absolute Error:', mae)
    print('Mean Squared Error:', mse)
    return mae, mse


# Считываем датасет и выводим все точки пересечения веса и роста
dataset = pd.read_csv('Davis.csv', index_col=0)
dataset = pd.DataFrame.dropna(dataset)
dataset.head()
dataset.describe()
dataset.plot(x='weight', y='height', style='bo')
plt.title('Dataset')
plt.xlabel('weight')
plt.ylabel('height')
plt.show()

# В x записываем значения веса, в y значение роста
X = dataset.iloc[:, 1:-3].values
y = dataset.iloc[:, 2:-2].values

# Разбиваем выборку на тестовую и обучаемую и обучаем
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


print('X: вес,  y: рост')
# Находим и печатаем  mae и mse
mae, mse = get_print_mae_mse(y_test, y_pred)

# Печатаем таблицу, где точки поделены на тестовые и обучаемые, так же выводим прямую регресии
plt.plot(X_train, y_train, 'bo', label='Train', alpha=0.2)
plt.plot(X_test, y_test, 'rx', label='Test', alpha=0.2)
line = np.arange(35, 120).reshape(-1, 1)
plt.plot(line, model.predict(line), 'g-')
plt.title('Result regression')
plt.xlabel('weight')
plt.ylabel('height')
plt.legend()
plt.show()

# В y также рост, а в x записываем рост, пол и repwt
y_2 = dataset.iloc[:, 2:-2].values
dataset = dataset.drop(['height', 'repht'], axis=1)
X_2 = dataset.iloc[:, :].values


# Если пол мужской, то ставим единицу, если женский то 0
for i in X_2:
    if i[0] == 'M':
        i[0] = 1
    elif i[0] == 'F':
        i[0] = 0

# Разбиваем выборку на тестовую и обучаемую и обучаем
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2)
model_2 = LinearRegression()
model_2.fit(X_train_2, y_train_2.reshape(-1, 1))
y_pred_2 = model_2.predict(X_test_2)

print('X: вес, пол, repwt, y: рост')
mae_2, mse_2 = get_print_mae_mse(y_test_2, y_pred_2)

print('\nDiff:')

print('Mean Absolute Error difference:', mae - mae_2)
print('Mean Squared Error difference:', mse - mse_2)
