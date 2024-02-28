import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("/Users/danteamicarella/Downloads/student-mat_modified (1)-1.csv")
df = df.drop(df.columns[0], axis=1)
print(df)

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

columns_drop = ["Performance", "school", "Pstatus", "nursery", "romantic", "sex"]
# Dropping based on Performance and unimportant columns
X = df.drop(columns_drop, axis = 1)
Y = df["Performance"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf_model = RandomForestClassifier()

# Perform Grid Search Cross-Validation to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, Y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(best_params)

# Train the model with the best parameters
best_rf_model = RandomForestClassifier(**best_params, random_state=42)
best_rf_model.fit(X_train, Y_train)

# Make predictions on the test set
y_pred = best_rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)

# Find features or columns that impact the model the most
feature_importances = best_rf_model.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
sorted_features = feature_importance_df.sort_values(by='Importance', ascending=False)
print(sorted_features)


