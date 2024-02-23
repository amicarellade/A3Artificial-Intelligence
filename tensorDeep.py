import pandas as pd
import numpy as np
# from tensorflow.keras import models, layers, utils, backend as K
import matplotlib.pyplot as plt
# import shap

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



# One layer 
# model = models.Sequential(name="Perceptron", layers=[
#     layers.Dense(             #a fully connected layer
#           name="dense",
#           input_dim=3,        #with 3 features as the input
#           units=1,            #and 1 node because we want 1 output
#           activation='linear' #f(x)=x
#     )
# ])
# model.summary()

