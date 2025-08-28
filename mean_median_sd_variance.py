from math import sqrt


arr = [174.2, 164, 170, 167.3, 180, 186, 178]

sum = 0
n = 0
for i in arr:
    sum += i
    n += 1

#Mean
mean = sum/n
print("Mean :",mean)

#Median
arr1 = sorted(arr)
n1 = int(n/2) 
med = arr1[n1]
print("Median :",med)

#Variance
Squ_total = 0
for i in arr:
    diff = i - mean
    squ = diff * diff
    Squ_total += squ

var = Squ_total/n
print("Variance :",var)

#Standard Daviation
sd = sqrt(var)
print("Standard Daviation :",sd)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = np.array([164, 167.3, 170, 174.2, 178, 180, 186])
plt.figure(figsize=(6,4))
sns.histplot(data, bins=5, kde=False, color='skyblue', edgecolor='black')
plt.title("Histogram")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()