import sys
import pandas as pd #для анализа и предобработки данных


from sklearn import ensemble #ансамбли
from sklearn import metrics #метрики
from sklearn import preprocessing #предобработка

from sklearn.model_selection import train_test_split #сплитование выборки

sys.path.insert(1, './')
from utils.functions import config_reader 

# Импортируем константы из файла config
config = config_reader('./config/config.json') 
#random_seed = config.seed_value

churn_data = pd.read_csv('./data/churn.zip')

# add new features
churn_data['BalanceSalaryRatio'] = churn_data['Balance']/churn_data['EstimatedSalary']
churn_data['TenureByAge'] = churn_data['Tenure']/(churn_data['Age'])
churn_data['CreditScoreGivenAge'] = churn_data['CreditScore']/(churn_data['Age'])
churn_data['Gender'] = churn_data['Gender'].apply(lambda x: 1 if x=='Male' else 0)
churn_data.head()


# Model for Germany
churn_data = churn_data[churn_data['Geography'] == 'Germany']


# drop not important features
churn_data = churn_data.drop(['Geography', 'RowNumber', 'CustomerId', 'Surname'], axis=1) #


X = churn_data.drop("Exited", axis=1)
y = churn_data["Exited"]
print('--------------start----------------')
print('Data shape :', X.shape)

# scaling data
scaler = preprocessing.MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

#  
#
#
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=config.random_seed) # для проверки - 0 для работы - random_seed

print('Train shape: {}'.format(X_train.shape))
print('Test  shape: {}'.format(X_test.shape))
print('\n')

# ratio
print('Train :\n', y_train.value_counts(normalize=True).round(2))
print('\n')
print('Test :\n', y_test.value_counts(normalize=True).round(2))
print('\n')

#Создаем объект класса случайный лес
rf = ensemble.RandomForestClassifier(
    n_estimators=500, #число деревьев
    criterion='entropy', #критерий эффективности
    max_depth=8, #максимальная глубина дерева
    min_samples_leaf = 10, # Минимальное число объектов в листе
    #max_features='sqrt', #число признаков из метода случайных подространств
    random_state=config.random_seed #генератор случайных чисел
)

#Обучаем модель 
rf.fit(X_train, y_train)

#Делаем предсказание класса для тренировочной выборки
y_pred_train = rf.predict(X_train)
#Выводим отчет о метриках
print('Train: {:.2f}'.format(metrics.f1_score(y_train, y_pred_train)))

#Делаем предсказание класса для тестовой выборки
y_pred_test = rf.predict(X_test)
#Выводим отчет о метриках
print('Test: {:.2f}'.format(metrics.f1_score(y_test, y_pred_test)))




#Считаем вероятности оттока клиентов модели случайный лес
y_test_proba_pred = rf.predict_proba(X_test)[:, 1]
#Для удобства завернем numpy-массив в pandas Series
y_test_proba_pred = pd.Series(y_test_proba_pred)

#Задаем оптимальный порог вероятностей
threshold_opt = 0.35
#Клиентов, для которых вероятность оттока > threshold относим к классу 1. В противном случае - к классу 0
y_pred_opt = y_test_proba_pred.apply(lambda x: 1 if x > threshold_opt else 0)
#Считаем метрики
#print(metrics.classification_report(y_test, y_pred_opt))
print('F1-score:', metrics.f1_score(y_test, y_pred_opt).round(2))