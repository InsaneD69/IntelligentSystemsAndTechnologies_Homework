from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import metrics


# Вычисляет ошибки,печатает матрицу ошибок и выводит значения accuracy, precision, recall, f1-score
def print_classif_result(y_test, y_pred):
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    print('Mean Absolute Error:', mae)
    print('Mean Squared Error:', mse)
    print('Confusion matrix:\n', metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))
    return mae, mse


X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=1,
    weights=(0.15, 0.85),
    class_sep=6.0,
    hypercube=False,
    random_state=2,
)

# Разбиваем выборку на тестовую и обучаемую
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Обучаем по методу логистической регрессии
model_logreg = LogisticRegression()
model_logreg.fit(X_train, y_train)
y_pred = model_logreg.predict(X_test)

# Обучаем по методу Kneighbors
model_KNeig = KNeighborsRegressor()
model_KNeig.fit(X_train, y_train)
y_pred2 = model_KNeig.predict(X_test)

for i in range(len(y_pred2)):
    if y_pred2[i] >= 0.5:
        y_pred2[i] = 1
    else:
        y_pred2[i] = 0

mae, mse = print_classif_result(y_test, y_pred)
mae_2, mse_2 = print_classif_result(y_test, y_pred2)

print('\nDiff:')

print('Mean Absolute Error difference:', mae - mae_2)
print('Mean Squared Error difference:', mse - mse_2)

# zero graph

plt.plot(X_train, y_train, 'bo', label='Train')
plt.plot(X_test, y_test, 'rx', label='Test')
plt.show()

# first graph

_, plts = plt.subplots()

precision_logreg, recall_logreg, _ = metrics.precision_recall_curve(y_test, y_pred)
precision_KNeig, recall_KNeig, _ = metrics.precision_recall_curve(y_test, y_pred2)
area_logreg = metrics.auc(recall_logreg, precision_logreg)
area_KNeig = metrics.auc(recall_KNeig, precision_KNeig)
plts.plot(recall_KNeig, precision_KNeig, color='gray')
plts.plot(recall_logreg, precision_logreg, color='black')
plts.set_title('Precision-Recall Curve')
plts.set_ylabel('Precision')
plts.set_xlabel('Recall')
plts.legend(handles=[
    mpatches.Patch(color='gray', label='PR KNearestNeighbors: %0.2f' % area_KNeig),
    mpatches.Patch(color='black', label='PR LogisticRegression: %0.2f' % area_logreg)
])
plt.show()

# second graph

y_pred_proba = model_logreg.predict(X_test)

fp_logreg, tp_logreg, _ = metrics.roc_curve(y_test, y_pred_proba)
roc_auc_logreg = metrics.auc(fp_logreg, tp_logreg)
plt.plot(fp_logreg, tp_logreg, color='black')
plt.ylabel('TP rate')
plt.xlabel('FP rate')

y_pred_proba = model_KNeig.predict(X_test)

fp_KNeig, tp_KNeig, _ = metrics.roc_curve(y_test, y_pred_proba)
roc_auc_KNeig = metrics.auc(fp_KNeig, tp_KNeig)
plt.plot(fp_KNeig, tp_KNeig, color='grey')
plt.ylabel('TP rate')

plt.legend(handles=[
    mpatches.Patch(color='grey', label='ROC KNearestNeighbors (area = %0.2f)' % roc_auc_KNeig),
    mpatches.Patch(color='black', label='ROC LogisticRegression (area = %0.2f)' % roc_auc_logreg)
])
plt.show()
