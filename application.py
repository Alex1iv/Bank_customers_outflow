import numpy as np
import pandas as pd 
import streamlit as st
from utils.reader_config import config_reader 
import pickle
import os



# Import of parameters
config = config_reader('config/config.json')

# loading saved model
with open(os.path.join(config.model_path, 'model_cv_opt.pkl'), 'rb') as f:
    rf_model = pickle.load(f)
    
genders_dict = {"Male": 1, "Female" : 0}
  
st.write("""
# This app predicts customer status!
""")


st.sidebar.header('User Input Parameters')

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

df = pd.DataFrame(data, index=[0]) # новое

df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else '0')

st.subheader('User Input parameters')
st.write(df.T)

# loaded model
rf=rf_model

# Prediction of probabilities:
y_new_proba_predict = rf.predict_proba(df)
print(y_new_proba_predict) #.round(2)


st.subheader('Predicted customer status:')
res = 'Exited' if np.argmax(y_new_proba_predict) == 1 else 'Loyal'
st.write('👉', res)

st.subheader('Probability of exit')
st.write(y_new_proba_predict[0][1].round(2)*100, '%')
