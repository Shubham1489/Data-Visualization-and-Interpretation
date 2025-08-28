import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("DATA_VISUALATION/stu_alc/student-mat.csv")

# Display data info
print(df.head())
print(df.info())


print(df.isnull().sum())

# # Numerical example
# df.fillna(df.mean(), inplace=True)

# # Categorical example
# df.fillna(df.mode()[0], inplace=True)