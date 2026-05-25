from sklearn import tree

#Rough = 1
#Smooth = 0

#Tennis = 1
#Cricket = 2


def main():
    print("Ball classification case study")

    #orignal encoded dataset
    # Independent variable
    X = [[35,1], [47, 1],[90, 0],[48, 1],[90, 0],[35,1],[92, 0],[35,1],[35,1],[35,1],[96, 0],[43, 1],[110, 0],[35, 1],[95, 0]]

    # Dependent variable
    Y = [1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2]

    #Independent variables for training
    Xtrain = [[35,1], [47, 1],[90, 0],[48, 1],[90, 0],[35,1],[92, 0],[35,1],[35,1],[35,1],[96, 0],[43, 1],[110, 0]]

    #Independent variables for testing
    Xtest = [[35, 1],[95, 0]]

    #Dependent variables for training
    Ytrain = [1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2]

    #Dependent variables for testing
    Ytest = [1, 2]

    modelobj = tree.DecisionTreeClassifier()        #object ahe

    trainedmodel = modelobj.fit(Xtrain,Ytrain)

    Result = trainedmodel.predict([[35,1]])          #1   2

    print(type(Result))

    if Result == 1:
        print("Object look like tennis ball")
    elif Result ==2:
        print("Object look like circket ball")
if __name__ == "__main__":
    main()