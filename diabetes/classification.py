import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

df = pd.read_csv("DATA_VISUALATION/diabetes/diabetes.csv")

# Display basic info
print(df.head())
print("\nDataset Shape:", df.shape)

# 1: Group by Target Variable
grouped = df.groupby("Outcome")
print("\nGroup counts:\n", grouped.size())

# 2: Divide dataset into two groups based on Outcome
group0 = df[df["Outcome"] == 0]
group1 = df[df["Outcome"] == 1]

# 3: Check Missing Values
print("\nMissing values per column:\n", df.isnull().sum())

cols_with_zero = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

print("\nMissing values after replacing 0s with NaN:\n", df.isnull().sum())

# 4: Handle Missing Values with Threshold
threshold = 0.2  # 20%

for col in df.columns[:-1]:  # except Outcome
    missing_percent = df[col].isnull().mean()
    if missing_percent > threshold:
        df = df.dropna(subset=[col])  # drop rows with too many missing values
    else:
        # Impute with K-Means clustering approach
        valid_rows = df.dropna()
        kmeans = KMeans(n_clusters=2, random_state=42)
        df.loc[:, col] = df[[col]].apply(
            lambda x: x.fillna(valid_rows[col].mean()) if x.isnull().any() else x
        )

print("\nMissing values after imputation:\n", df.isnull().sum())

# 6: Compute t-test and p-values
p_values = {}
for col in df.columns[:-1]:  # exclude Outcome
    t_stat, p_val = ttest_ind(group0[col], group1[col], nan_policy='omit')
    p_values[col] = p_val

# 7: Rank Features by p-values
sorted_features = sorted(p_values.items(), key=lambda x: x[1])
print("\nFeatures ranked by p-value:")
for feat, p in sorted_features:
    print(f"{feat}: p={p:.5f}")

# 8: Classification
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# 9: Evaluation Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# 2. ROC & AUC
y_prob = classifier.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

# 3. Classification Report (convert to DataFrame for plotting)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Dashboard Plot
fig, axes = plt.subplots(1, 3, figsize=(18,5))

# ---- (A) Confusion Matrix ----
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["0","1"], yticklabels=["0","1"])
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# ---- (B) ROC Curve ----
axes[1].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
axes[1].plot([0,1], [0,1], linestyle="--", color="gray")
axes[1].set_title("ROC Curve")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend(loc="lower right")

# ---- (C) Classification Report ----
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="Greens", fmt=".2f", ax=axes[2])
axes[2].set_title("Classification Report (Precision/Recall/F1)")

plt.tight_layout()
plt.show()
