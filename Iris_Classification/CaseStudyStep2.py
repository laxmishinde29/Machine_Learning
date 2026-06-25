import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

Border = "-"*40
##############################################################
# Step 1: Load the Dataset
##############################################################

print(Border)
print("Step 1 : Load the dataset")
print(Border)

DatasetPath = "iris.csv"

df = pd.read_csv(DatasetPath)

print("Dataset gets loaded succesfully...")
print("Initial entries from dataset :")
print(df.head())

##############################################################
# Step 2: Data Anylysis(EDA Exploritory data analysis)
##############################################################

print(Border)
print("Step 1 : Data Analysis")
print(Border)

print("Shape of Dataset :",df.shape)
print("Column names :",list(df.columns))

print("Missing values (per column)")
print(df.isnull().sum())

print("Class Distribution (Species count)")
print(df["species"].value_counts())

print("Statistical Report of dataset")
print(df.describe())