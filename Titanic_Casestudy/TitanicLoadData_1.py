import pandas as pd 
import numpy as np 
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#----------------------------------------------------
# Function name :   DisplayInfo
# Description :     It display the formated title
# Parameters :      title (str)
# Return :          None 
# Data :            14/03/2026
# Author :          Laxmi Nandkumar Shinde
#----------------------------------------------------

def DisplayInfo(Title):
    print("\n"+"="*70)
    print(Title)
    print("="*70)

#----------------------------------------------------
# Function name :   ShowData
# Description :     It show basic information about dataset
# Parameters :      Dataset (df)
#                   df =>   pandas dataframe object
#                   message
#                   message =>  Heading text to display
# Return :          None 
# Data :            14/03/2026
# Author :          Laxmi Nandkumar Shinde
#----------------------------------------------------

def ShowData(df,message):
    DisplayInfo(message)

    print("First 5 rows of dataset")
    print(df.head())

    print("\n Shape of dataset")
    print(df.shape)

    print("\n Columns name")
    print(df.columns.tolist())

    print("Missing value in each columns")
    print(df.isnull().sum())

#----------------------------------------------------
# Function name :   MarvellousTitanicLogistic
# Description :     This is main pipeline controller
#                   It load a dataset and show raw data 
#                   It preprocess dataset and train model
# Parameters :      Data Path of dataset file
# Return :          None 
# Data :            14/03/2026
# Author :          Laxmi Nandkumar Shinde
#----------------------------------------------------

def MarvellousTitanicLogistic(Datapath):
    DisplayInfo("Step 1 : Loading the dataset")

    df = pd.read_csv(Datapath)

    ShowData(df,"Initial dataset")
#----------------------------------------------------
# Function name :   main
# Description :     Starting point application
# Parameters :      None
# Return :          None 
# Data :            14/03/2026
# Author :          Laxmi Nandkumar Shinde
#----------------------------------------------------

def main():
    MarvellousTitanicLogistic("MarvellousTitanicDataset.csv")

if __name__ == "__main__":
    main()