import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("DATA_VISUALATION/diabetes/diabetes.csv")

# Display data info
print(df.head())
print(df.info())

column = df['Glucose']

print(df.isnull().sum())

# Numerical example
df.fillna(df.mean(), inplace=True)

# Categorical example
# column.fillna(column.mode()[0], inplace=True)

plt.hist(column, bins=10, edgecolor='black')
plt.title('Histogram of Glucose')
plt.xlabel('Glucose')
plt.ylabel('Frequency')
plt.show()


Q1 = column.quantile(0.25)
Q3 = column.quantile(0.75)
IQR = Q3 - Q1

outliers = column[(column < Q1 - 1.5 * IQR) | (column > Q3 + 1.5 * IQR)]
print(outliers)

column_no_outliers = column[(column >= Q1 - 1.5 * IQR) & (column <= Q3 + 1.5 * IQR)]

plt.hist(column_no_outliers, bins=10, edgecolor='black')
plt.title('Histogram after removing outliers')
plt.xlabel('Glucose')
plt.ylabel('Frequency')
plt.show()

sns.pairplot(df)
plt.show()