import pandas as pd 
import numpy as np 
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#----------------------------------------------------
# Function name :   Load Preserve model
# Description :     It is used to preserve model
# Parameters :      model, filename
# Return :          None 
# Data :            14/03/2026
# Author :          Laxmi Nandkumar Shinde
#----------------------------------------------------
def LoadPreseveModel(filename):
    loaded_model = joblib.load(filename)
    print("Model succesfully loaded ")

    return loaded_model

#----------------------------------------------------
# Function name :   Preserve model
# Description :     It is used to preserve model on secondary 
# Parameters :      model, filename
# Return :          None 
# Data :            14/03/2026
# Author :          Laxmi Nandkumar Shinde
#----------------------------------------------------

def PreseveModel(model,filename):
    joblib.dump(model,filename)

    print("preserved model sucessfully:",filename)

#----------------------------------------------------
# Function name :   TrainTitanicModel
# Description :     It does split X,Y,training data, testing data
# Parameters :      df
# Return :          None 
# Data :            14/03/2026
# Author :          Laxmi Nandkumar Shinde
#----------------------------------------------------

def TrainTitanicModel(df):
    #split feature and labels
    X = df.drop("Survived", axis = 1)
    Y = df["Survived"]

    print("\nFeatures :")
    print(X.head())

    print("\nlabels:")
    print(Y.head())

    print("Shape of X :",X.shape)
    print("Shape of Y :",Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)
    print("X_train shape:",X_train.shape)
    print("X_test shape:",X_test.shape)
    print("Y_train shape:",Y_train.shape)
    print("Y_test shape:",Y_test.shape)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train,Y_train)

    print("Model trained succesfully")

    print("\n Intercept of model :")
    print(model.intercept_)

    print("\n Coefficient of model")
    for feature, coefficent in zip(X.columns, model.coef_[0]):
        print(feature, ":", coefficent)

    PreseveModel(model,"Marvelloustitanic.pkl")

    loaded_model = LoadPreseveModel("Marvelloustitanic.pkl")

    Y_pred = loaded_model.predict(X_test)

    accuracy = accuracy_score(Y_pred,Y_test)

    print("Accuracy is :",accuracy)

    cm = confusion_matrix(Y_pred, Y_test)
    print("Confusin matrix is :")
    print(cm)

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
# Function name :   CleanTitanicData
# Description :     It does preprocesing 
#                   It removed unneccessary columns
#                   It handales missing values
#                   It converts text data to numeric format 
#                   It does encoding to categorical columns
# Parameters :      df -> pandas dataframe
# Return :          df -> clean pandas dataframe 
# Data :            14/03/2026
# Author :          Laxmi Nandkumar Shinde
#----------------------------------------------------

def CleanTitanicData(df):
    DisplayInfo("Step 2 : Original data")
    print(df.head())

    #Remove unnessary columns 
    drop_columns = ["Passengerid","zero","Name","Cabin"]
    existing_columns = [col for col in drop_columns if col in df.columns]

    print("\n columns to be droped :")
    print(existing_columns)

    #drop the unwanted columns 
    df = df.drop(columns = existing_columns)
    DisplayInfo("Step 2 : Data after call removeal")
    print(df.head())

    #Handle age column 
    if "Age" in df.columns:
        print("Age column before filling missing values")
        print(df["Age"].head(10))

        # coerce -> Invalid values gets converted as NaN 
        df["Age"] = pd.to_numeric(df["Age"],errors = "coerce")

        age_median = df["Age"].median()

        #replace missing values with median
        df["Age"] = df["Age"].fillna(age_median)

        print("Age median is:",age_median)

        print("\n Age column after preprocesing")
        print(df["Age"].head(10))
        print("Age median is:",age_median)

    #Handel fare column 
    if "Fare" in df.columns:
        print("\n Fare column before preprocessing")
        print(df["Fare"].head(10))

        df["Fare"] = pd.to_numeric(df["Fare"],errors = "coerce")

        Fare_median = df["Fare"].median()

        print("\n median of Fare column is :",Fare_median)

        #replace missing values with median
        df["Fare"] = df["Fare"].fillna(Fare_median)

        print("\n Fare column after preprocesing")
        print(df["Fare"].head(10))

    # Handel Embarked column (direction assign karta east,.. )
    if "Embarked" in df.columns:
        print(df["Embarked"].head(10))

        #convert the data into string
        df["Embarked"] = df["Embarked"].astype(str).str.strip()

        #Remove missing values 
        df["Embarked"] = df["Embarked"].replace(['nan','None',''],np.nan)

        #get most frequent value
        embarked_mode = df["Embarked"].mode()[0]
        print("Mode of embarked column :",embarked_mode)

        df["Embarked"] = df["Embarked"].fillna(embarked_mode)

        print("\n Embarked column after preprocesing")
        print(df["Embarked"].head(10))

    #Handel fare column 
    if "Sex" in df.columns:
        print("\n Sex column before preprocessing")
        print(df["Sex"].head(10))

        df["Sex"] = pd.to_numeric(df["Sex"],errors = "coerce")

        print("\n Sex column after preprocesing")
        print(df["Sex"].head(10))

    DisplayInfo("Data after preprocessing")
    print(df.head())

    print("\n Missing value of after preprocesing")
    print(df.isnull().sum())

    #Encode Embarked Column 
    df = pd.get_dummies(df, columns=["Embarked"],drop_first = True)
    print("\n Data after encoding")

    print(df.head())
    print("Shape of dataset : ",df.shape)

    #convert boolean columns into integer
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    print("\n Data after encoding")

    print(df.head())

    return df


#interview = what is onehot encoding
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

    df = CleanTitanicData(df)

    TrainTitanicModel(df)

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