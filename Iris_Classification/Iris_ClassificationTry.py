import pandas as pd
#from sklearn.datasets import load_iris


def main():
    print("Iris classification case study")

    Dataset = pd.read_csv("iris.csv")

    Border = "-"*40

    print(Border)

    X = Dataset.iloc[:, :-1]   # All columns except last
    y = Dataset.iloc[:, -1]    # Last column (label)

    for i in range(len(Dataset)):
        print("ID %d, Features %s, Label %s" %(i, list(X.iloc[i]), y.iloc[i]))
        #print("ID %d, Features %s" %(i, Dataset.data[i]))

    print(Border)

if __name__ == "__main__":
    main()