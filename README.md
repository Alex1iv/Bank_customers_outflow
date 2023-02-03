# Bank customers outflow
---

## Content

[1. Summary](README.md#Summary)   
[2. Data and methods](README.md#Data-and-methods)     
[3. Project structure](README.md#Project-structure)


### Summary
It was predicted the client outflow from a certain bank using the ensamble ML model. It was created a micro service application to calculate the F1-score. Several model were compared and the Random Forest model showed satisfactory result (F1-score=0.82) even after reducing the number of features from 11 to 4 ()


### Data and methods
It is wiedly known that retaintion of an existing client is cheaper than to find a new one. Thus, the client, an international bank, wants to predict whether the client is going to leave it or he is still loyal to the bank. If yes, it will be offered some additional options to recover his loyalty to the company services.

The assignment is narrowed do the binary classification: whether or not the client leave the bank.

The clients data was taken from the [Kaggle.com](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers) website.

In order to identify the best model, it was tested several algorythms: 
* linear regression, 
* polinominal regression with L1 and L2 regulirization, 
* decision tree 
* random forest

It was found that classes in the dataset are not balanced. Taking this fact into account, it was decided to use F1-score metrics since it provided harmonic mean of two other metrics: precision and recall.



:arrow_up:[ to content](_)

## Project structure

<details>
  <summary>display project structure </summary>

```Python
Bank_customers_outflow
├── .gitignore
├── config              # configuration parameters
│   └── config.json     
├── data                # data archive
│   └── churn.zip      
├── figures             # figures
│   ├── fig_1.png
......
│   └── fig_9.png
├── models              # models storage
│   ├── models.py
│   └── __ init __.py
├── notebooks           # project notebooks storage
│   └── Bank_clients_en.ipynb
├── README.md
└── utils
    ├── functions.py
    ├── reader_config.py
    └── __ init __.py

```
</details>