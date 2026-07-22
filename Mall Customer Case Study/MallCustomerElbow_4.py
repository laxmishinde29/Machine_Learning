import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans

def main():
    #-----------------------------------------------
    #Step 1 : Load the dataset
    #-----------------------------------------------

    print("Step 1 : Load the dataset")
    df = pd.read_csv("Mall_Customers.csv")

    print("First few records :")
    print(df.head())

    print("Shape of datatset:")
    print(df.shape)

    print("Missing values:")
    print(df.isnull().sum())

    #-----------------------------------------------
    #Step 2 : Select features(Independent)
    #-----------------------------------------------

    print("Step 2 : Select features(Independent)")

    X = df[["AnnualIncome", "SpendingScore"]]
    print("Selected features :")
    print(X.head())

    print("Shape of selected features :")
    print(X.shape)


    #-----------------------------------------------
    #Step 3 : Scale the data
    #-----------------------------------------------

    print("Step 3 : Scale the data")

    scalar = StandardScaler()
    X_Scaled = scalar.fit_transform(X)

    print("Data after scalling :")
    print(X_Scaled[:5])

    #-----------------------------------------------
    #Step 4 : use eblow methos
    #-----------------------------------------------

    print("Step 4 : use eblow methos")

    WCSS = []

    for i in range (1,11):
        model = KMeans(n_clusters = i, random_state = 42, n_init = 10)
        model.fit(X_Scaled)
        WCSS.append(model.inertia_)

    plt.figure(figsize = (8,5))
    plt.plot(range(1,11), WCSS, marker = 'o')
    plt.xlabel("Number of cluster")
    plt.ylabel("WCSS")
    plt.title("Elbow method")
    plt.grid(True)
    plt.show()

    #-----------------------------------------------
    #Step 5 : train the model
    #-----------------------------------------------

    print("Step 5 : train the model")

    model = KMeans(n_clusters = 4, random_state=42, n_init = 10)
    clusters = model.fit_predict(X_Scaled)

    df["clusters"] = clusters

    print("Dataset with cluster")
    print(df.head(30))


if __name__ == "__main__":
    main()