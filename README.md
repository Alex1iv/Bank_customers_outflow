# Bank customers outflow
---

## Content

[1. Summary](README.md#Project-description)
[2. Data](README.md#Data)                               
[3. Inferences](README.md#Inferences)                   


### Summary
It was predicted the client outflow from a certain bank using the ensamble ML model. The model quality is estimated using the F1-metrics.


### Data and methods
It is wiedly known that retaintion of an existing client is cheaper than to find a new one. Thus, the client, an international bank, wants to predict whether the client is going to leave it or he is still loyal to the bank. If yes, it will be offered some additional options to recover his loyalty to the company services.

The assignment is narrowed do the binary classification: whether or not the client leave the bank.

The clients data was taken from the [Kaggle.com](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers) website.

In order to identify the best model, it was tested several algorythms: 
* linear regression, 
* polinominal regression, 
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
    └── __ init __.py

```
</details>