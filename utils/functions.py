import pandas as pd
import sys
sys.path.insert(1, '../')

class My_Dataframe():
    def __init__(self, data, sep):
        self.df = pd.read_csv(data, sep=',')
    

    
churn_data = My_Dataframe('../data/churn.zip', sep=',')    

print(churn_data.shape)