import pandas as pd
import numpy as np
import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import string
import random
import string
from sklearn import linear_model
import os
import matplotlib.pyplot as plt


# +
# load the data
def readFiles(file):
    file_path = os.path.join('data', file)
    return pd.read_csv(file_path, compression='gzip')

trainData = readFiles('interactions_train.csv.gz')
validData = readFiles('interactions_validation.csv.gz')
testData = readFiles('interactions_test.csv.gz')
recipes = readFiles('RAW_recipes.csv.gz')
interactions = readFiles('RAW_interactions.csv.gz')
# -

trainData.head()

print(trainData.head().to_markdown(index=False))

recipes.head()

# +
# Exploratory Data Analysis
# -

# Display basic information
print("Recipes Data Info:")
print(recipes.info())

print("\nInteractions Data Info:")
print(interactions.info())


print(recipes.describe().to_markdown())

print(interactions.describe().to_markdown())

# +
# Check for missing values
print("Missing Values in Recipes Data:")
print(recipes.isnull().sum())

print("\nMissing Values in Interactions Data:")
print(interactions.isnull().sum())

# +
# Unique values
print("Unique Values in Key Columns - Recipes:")
print(recipes.nunique())

print("\nUnique Values in Key Columns - Interactions:")
print(interactions.nunique())
# -

# Plot distributions of numeric columns in recipes
recipes.hist(figsize=(12, 8), bins=20)
plt.suptitle("Distribution of Numerical Features in Recipes Data")
plt.show()

# Plot distributions of numeric columns in interactions
interactions.hist(figsize=(12, 8), bins=20)
plt.suptitle("Distribution of Numerical Features in Interactions Data")
plt.show()

# Most common ingredients:
print("Most Common Ingredients:")
print(recipes['ingredients'].value_counts().head(10))

# Distribution of ratings
print("Rating Distribution:")
print(interactions['rating'].value_counts())
interactions['rating'].value_counts().plot(kind='bar')
plt.title("Rating Distribution")
plt.show()


