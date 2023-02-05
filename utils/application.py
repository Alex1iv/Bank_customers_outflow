import numpy as np
import pandas as pd 
import streamlit as st
from utils.reader_config import config_reader 
from models.models_collection import ModelRandomForest

from sklearn import preprocessing  
from sklearn.model_selection import train_test_split 


# Import of parameters
config = config_reader('config/config.json')
    
    
st.write("""
# This app predicts customer status!
""")

genders_dict = {
    "Male": 1,
    "Female" : 0
}


st.sidebar.header('User Input Parameters')


def user_input_features():
    
    
    # Tenure
    Tenure = int(st.sidebar.number_input("Tenure", min_value=0, max_value=10, value=0, step=1, help="Tenure, years"))
    
    Credit_score = st.sidebar.slider('CreditScore', 350, 850, 750)
    Gender = st.sidebar.selectbox("Pick genders", genders_dict.keys(), help="Male - 1, Female - 0.")
    Age = st.sidebar.slider('Age', 18, 80, 20)
    Balance = st.sidebar.slider('Balance', 28000, 210000, 70000)
    NumOfProducts = int(st.sidebar.number_input("Number of products", min_value=1, max_value=4, value=1, step=1, help="Number of products"))
    
    HasCreditCard = st.sidebar.slider('HasCreditCard', 0, 1, 0)
    IsActiveMember = st.sidebar.slider('IsActiveMember', 0, 1, 1)
    Salary = st.sidebar.slider('Salary', 0, 200000, 70000)
    
    st.sidebar.markdown("---")
    
    data = {
        'Credit_score': Credit_score,
        'Gender': Gender,
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': HasCreditCard,
        'IsActiveMember': IsActiveMember,
        'EstimatedSalary': Salary,
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else '0')

st.subheader('User Input parameters')
st.write(df.T)

# Import of parameters
#config = config_reader('config/config.json')

    
churn_data = pd.read_csv('data/churn.zip')
# drop not important features
churn_data = churn_data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1) 

# add new features
#churn_data['BalanceSalaryRatio'] = churn_data['Balance']/churn_data['EstimatedSalary']
#churn_data['TenureByAge'] = churn_data['Tenure']/(churn_data['Age'])
#churn_data['CreditScoreGivenAge'] = churn_data['CreditScore']/(churn_data['Age'])
churn_data['Gender'] = churn_data['Gender'].apply(lambda x: 1 if x=='Male' else 0)
churn_data.head()


# Model for Germany
churn_data = churn_data[churn_data['Geography'] == 'Germany']

# Del feature Geography
churn_data = churn_data.drop(['Geography'], axis=1) 

print('Data shape :', churn_data.shape)

X, y = churn_data.drop("Exited", axis=1), churn_data["Exited"]


# scaling data
scaler = preprocessing.MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# split the data using stratification
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=0)  #config.random_seed

# Train model -------------------------------------
rf = ModelRandomForest(config)

rf.fit(X_train, y_train)

# Predicting
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

# Calculation the client outflow probability for the test sample
y_test_proba_pred = rf.predict_proba(X_test)[:, 1]

y_pred_opt = np.where(y_test_proba_pred > 0.35, 1, 0) 


#Делаем предсказание вероятностей:
y_new_proba_predict = rf.predict_proba(df)
print(y_new_proba_predict) #.round(2)


st.subheader('Predicted customer status:')
res = 'Loyal' if np.argmax(y_new_proba_predict) == 1 else 'Exited'
st.write('👉', res)

st.subheader('Probability of exit')
st.write(y_new_proba_predict[0][1].round(2)*100, '%')
