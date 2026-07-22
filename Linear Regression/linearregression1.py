import numpy as np 
import pandas as pd
import matplotlib as plt

def marvellousPredictor():
    #load the data 
    X = [1,2,3,4,5]
    Y = [3,4,2,4,5]

    print("Values of independent variable : X-",X)
    print("Values of dependent variable : Y-",Y)

    mean_X = np.mean(X)
    mean_y = np.mean(Y)

    print("X_MEAN is :",mean_X)
    print("Y MEAN is :",mean_y)

def main():
    marvellousPredictor()

if __name__ == "__main__":
    main()
