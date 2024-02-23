import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/danteamicarella/Downloads/student-mat_modified (1)-1.csv")
df = df.drop(df.columns[0], axis=1)
# print(df)

# Convert object into indexes
category_columns = list()
for column in df.columns:
    if df[column].dtype == 'object':
        category_columns.append(column)

print(category_columns)

mapping_functions = dict()
for column in category_columns:
    values = df[column].unique()
    mapping_function = dict()
    for value_idx, value in enumerate(values):
        mapping_function[value] = value_idx
    mapping_functions[column] = mapping_function

print(mapping_functions)

for column in category_columns:
    df[column] = df[column].map(mapping_functions[column])

print(df.head())



