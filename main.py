import numpy as np
import pandas as pd #для анализа и предобработки данных

from utils.functions import config_reader 
from models.models_collection import Model_RandomForest


from sklearn import preprocessing #предобработка
from sklearn.model_selection import train_test_split #сплитование выборки
from sklearn import metrics #метрики


# Импортируем константы из файла config

config = config_reader('./config/config.json') 

churn_data = pd.read_csv('data/churn.zip')

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
print('Data shape :', churn_data.shape)

X = churn_data.drop("Exited", axis=1)
y = churn_data["Exited"]

print('--------------start----------------')


# scaling data
scaler = preprocessing.MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=0) # для проверки - 0 для работы - config.random_seed

print('Train shape: {}, Test  shape: {}'.format(X_train.shape, X_test.shape), '\n')

# ratio
print('Train :\n{}, Test :\n{}'.format(y_train.value_counts(normalize=True).round(2), y_test.value_counts(normalize=True).round(2)), '\n')


rf = Model_RandomForest(config=config)

rf.fit(X_train, y_train)

# #Делаем предсказание класса для тренировочной выборки
# y_pred_train = rf.predict(X_train)
# #Выводим отчет о метриках
# print('Train: {:.2f}'.format(metrics.f1_score(y_train, y_pred_train)))

# #Делаем предсказание класса для тестовой выборки
# y_pred_test = rf.predict(X_test)
# #Выводим отчет о метриках
# print('Test: {:.2f}'.format(metrics.f1_score(y_test, y_pred_test)))

# Optimization of the model by F1-score-------------------------------------

#Считаем вероятности оттока клиентов модели случайный лес на тестовой выборке
y_test_proba_pred = rf.predict_proba(X_test)[:, 1]


f1_scores = []
#Сгенерируем набор вероятностных порогов в диапазоне от 0.1 до 1
thresholds = np.arange(0.1, 1, 0.05)
#В цикле будем перебирать сгенерированные пороги
for threshold in thresholds:
    #Клиентов, для которых вероятность оттока > threshold относим к классу 1. В противном случае - к классу 0
    y_test_pred = np.where(y_test_proba_pred>threshold, 1, 0)
    #Считаем метрику и добавляем их в списки
    f1_scores.append(metrics.f1_score(y_test, y_test_pred))
    
# Maximal F1-score
max_f1_score = max(f1_scores)

# transform list of f1-scores to array
f1_scores = np.array(f1_scores)

# The probability value of the best f1-score value
threshold_opt = round(0.05 + (np.argmax(f1_scores) + 1)*0.05, 2)
print(f'Best F1-score: {max_f1_score.round(3)}, Maximal probability: {threshold_opt}')

#Задаем оптимальный порог вероятностей
threshold_opt = threshold_opt
#Клиентов, для которых вероятность оттока > threshold относим к классу 1. В противном случае - к классу 0
y_pred_opt = np.where(y_test_proba_pred > threshold_opt, 1, 0) 
#Считаем метрики
#print(metrics.classification_report(y_test, y_pred_opt))
print('F1-score:', metrics.f1_score(y_test, y_pred_opt).round(3))