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

    print("X_MEAN is :",mean_X) #3.0
    print("Y_MEAN is :",mean_y) #3.6

    n = len(X)  #5

    # Y = mX + C

    # m = (summ(X-X_bar)*(Y-Y_bar)) / (summ(X-X_bar) **2)

    numerator = 0
    denominator = 0

    for i in range(n):
        numerator = numerator + ((X[i] - mean_X) * (Y[i] - mean_y))
        denominator = denominator + ((X[i] - mean_X) ** 2)

    m = numerator / denominator

    print("Slop of line that is m :",m)     #0.4

    C = mean_y - (m * mean_X)

    print("Y intercept of line that is C :",C)

def main():
    marvellousPredictor()

if __name__ == "__main__":
    main()