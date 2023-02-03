import numpy as np
import pandas as pd 

from utils.reader_config import config_reader 
from models.models_collection import ModelRandomForest


from sklearn import preprocessing  
from sklearn.model_selection import train_test_split 
from sklearn import metrics  



# Import of parameters
config = config_reader('config/config.json')

def get_data():
     
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

    # split the data using stratification
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=0) # для проверки - 0 для работы - config.random_seed

    print('Train shape: {}, Test  shape: {}'.format(X_train.shape, X_test.shape), '\n')

    # ratio
    print('Train :\n{}, Test :\n{}'.format(y_train.value_counts(normalize=True).round(2), y_test.value_counts(normalize=True).round(2)), '\n')


    rf = ModelRandomForest(config)

    rf.fit(X_train, y_train)

    # Optimization of the model by F1-score-------------------------------------

    # Calculation the client outflow probability for the test sample
    y_test_proba_pred = rf.predict_proba(X_test)[:, 1]

    # Creartion of an empty list to store f1 metrics
    f1_scores = []
    # Generation of an array of treshholds 
    thresholds = np.arange(0.1, 1, 0.05)
    
    # cycle to iterate on each treshhold
    for threshold in thresholds:
        # clients with the outflow probability > threshold belong to the class 1 else class 0
        y_test_pred = np.where(y_test_proba_pred>threshold, 1, 0)
        
        # calculation of metrics
        f1_scores.append(metrics.f1_score(y_test, y_test_pred))
        
    # Maximal F1-score
    max_f1_score = max(f1_scores)

    # transform list of f1-scores to array
    f1_scores = np.array(f1_scores)

    # The probability value of the best f1-score value
    threshold_opt = thresholds[np.argmax(f1_scores)].round(3)
    
    print(f'Best F1-score: {max_f1_score.round(3)}, Maximal probability: {threshold_opt}')

    # Set an optimal probability value
    threshold_opt = threshold_opt
    # Clients with the outflow probability > threshold belong to the class 1 else class 0
    y_pred_opt = np.where(y_test_proba_pred > threshold_opt, 1, 0) 
    
    # calculation of F1 metric
    f1_scores_total = metrics.f1_score(y_test, y_pred_opt).round(3)
    
    return f1_scores_total