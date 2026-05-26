import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def MarvellousClassifier(DataPath):
    Border = "-"*40

    #step 1 : Load the dataset from csv file
    print(Border)
    print("step 1 : Load the dataset from csv file")
    print(Border)

    df = pd.read_csv(DataPath)
    print(Border)
    print("Some entries from dataset")
    print(df.head())
    print(Border)

    #step2 : clean the dataset by removing empty rows
    print(Border)
    print("step2 : clean the dataset by removing empty rows")
    print(Border)

    df.dropna(inplace = True)
    print("Total records :",df.shape[0])
    print("Total columns :",df.shape[1])
    print(Border)

    #step3 : seprate independent and dependent variables
    print(Border)
    print("step3 : seprate independent and dependent variables")
    print(Border)

    X = df.drop(columns=['Class'])
    Y = df['Class']

    print("Shape of X :",X.shape)
    print("Shape of Y :",Y.shape)

    print(Border)
    print("Input columns :",X.columns.tolist())
    print("Output column : Class")

    #step 4 : split the dataset for training or testing
    print(Border)
    print("step 4 : split the dataset for training or testing")
    print(Border)

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42,stratify= Y)

    print(Border)
    print("Information of training and testing data")
    print("X_train shape :",X_train.shape)
    print("X_test shape :",X_test.shape)
    print("Y_train shape :",Y_train.shape)
    print("Y_test shape :",Y_test.shape)

    print(Border)

    #step 5 : Feature scaling
    print(Border)
    print("step 4 : Feature scaling")
    print(Border)

    scalar = StandardScaler()
    #independent variable scaling
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.fit_transform(X_test)

    print("Feature scaling is done")

    #step 6: Explore the multiple values of k
    #Hyperparameter tunning (k)

    accuracy_scores= []
    K_values = range(1,21) 

    for k in K_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled,Y_train)
        model.predict 
        Y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(Y_test,Y_pred)
        accuracy_scores.append(accuracy)

    print(Border)
    print("Acuracy report of all k values from 1 to 20")
    for value in accuracy_scores:
        print(value)

    print(Border)


def main():
    Border = "-"*40

    print(Border)
    print("Wine Classifier using KNN")
    print(Border)

    MarvellousClassifier("WinePredictor.csv")

if __name__ == "__main__":
    main()

