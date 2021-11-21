import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Чтение набора данных
dataset = pd.read_csv('creditcard.csv')
X = dataset.iloc[:, 0:30].values
y = dataset.iloc[:, 30].values

# Установление размера выборки
train_size = 0
while train_size <= 0 or train_size >= 100:
    train_size = float(input("Введите процент обучающей выборки --> "))
    train_size /= 100.0

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

# Трансформация выборки
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Тренировка модели
print('Обучение модели, пожалуйста подождите...')
cls = RandomForestClassifier(n_estimators=100, random_state=0)
cls.fit(X_train, y_train)
print('Обучение модели прошло успешно')

# Предсказание модели
y_pred = cls.predict(X_test)
print('Проверка модели прошла успешно')

print('\nПолученный результат:\n')
print(classification_report(y_test, y_pred))
print("Точность: ")
print(accuracy_score(y_test, y_pred))
