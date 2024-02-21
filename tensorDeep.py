import pandas as pd
import numpy as np
from tensorflow.keras import models, layers, utils, backend as K
import matplotlib.pyplot as plt
import shap

df = pd.read_csv("/Users/danteamicarella/Downloads/student-mat_modified (1)-1.csv")
df = df.drop(df.columns[0], axis=1)
print(df)


