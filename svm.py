import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import svm

df = pd.read_csv("/Users/danteamicarella/Downloads/student-mat_modified (1)-1.csv")
df = df.drop(df.columns[0], axis=1)
# print(df)

# Convert object into indexes
category_columns = list()
for column in df.columns:
    if df[column].dtype == 'object':
        category_columns.append(column)

# print(category_columns)

mapping_functions = dict()
for column in category_columns:
    values = df[column].unique()
    mapping_function = dict()
    for value_idx, value in enumerate(values):
        mapping_function[value] = value_idx
    mapping_functions[column] = mapping_function

# print(mapping_functions)

for column in category_columns:
    df[column] = df[column].map(mapping_functions[column])

# print(df.head())

# Labeling categorical columns
for column in category_columns:
    df[column] = df[column].astype('category')

# Feature Selection
columns_drop = ["Performance", "schoolsup", "higher", "famsup", "nursery", "school", "famrel", "Fedu"]
# columns_drop = ["Performance"]

X = df.drop(columns_drop, axis = 1)
Y = df["Performance"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)

clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))

# Train the model
clf.fit(X_train, Y_train)

coefficients = clf.named_steps['svc'].coef_

# Map coefficients to feature names
feature_names = X_train.columns
coef_dict = {}
for feature_name, coef in zip(feature_names, coefficients[0]):
    coef_dict[feature_name] = coef

# Sort the coefficients by their absolute values to identify the most impactful features
sorted_coef = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)

for feature, coef in sorted_coef:
    print(f"{feature}: {coef}")

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)