# ==========================================================
# TITANIC SURVIVAL PREDICTION USING LOGISTIC REGRESSION
# ==========================================================

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================================
# STEP 1 : Load Dataset
# ==========================================================

df = pd.read_csv("Titanic-Dataset.csv")

print("=" * 60)
print("First Five Rows")
print("=" * 60)
print(df.head())

print("\nDataset Shape:", df.shape)

# ==========================================================
# STEP 2 : Dataset Information
# ==========================================================

print("\n")
print("=" * 60)
print("Dataset Information")
print("=" * 60)
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# ==========================================================
# STEP 3 : Handle Missing Values
# ==========================================================

# Fill Age with Median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked with Most Frequent Value
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin Column
df.drop('Cabin', axis=1, inplace=True)

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# ==========================================================
# STEP 4 : Convert Categorical Columns
# ==========================================================

encoder = LabelEncoder()

df['Sex'] = encoder.fit_transform(df['Sex'])

df['Embarked'] = encoder.fit_transform(df['Embarked'])

# ==========================================================
# STEP 5 : Drop Unnecessary Columns
# ==========================================================

df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

print("\nProcessed Dataset")
print(df.head())

# ==========================================================
# STEP 6 : Data Visualization
# ==========================================================

plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Gender vs Survival")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Passenger Class vs Survival")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ==========================================================
# STEP 7 : Prepare Data
# ==========================================================

X = df.drop('Survived', axis=1)

y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# STEP 8 : Train Logistic Regression Model
# ==========================================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# ==========================================================
# STEP 9 : Prediction
# ==========================================================

y_pred = model.predict(X_test)

# ==========================================================
# STEP 10 : Model Evaluation
# ==========================================================

print("\n")
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy*100:.2f}%")

print("\nConfusion Matrix")

cm = confusion_matrix(y_test, y_pred)

print(cm)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=["Not Survived","Survived"],
    yticklabels=["Not Survived","Survived"]
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

print("\nClassification Report")

print(classification_report(y_test, y_pred))

# ==========================================================
# STEP 11 : Predict New Passenger
# ==========================================================

print("\n")
print("=" * 60)
print("Prediction for New Passenger")
print("=" * 60)

new_passenger = pd.DataFrame({
    'Pclass':[3],
    'Sex':[1],
    'Age':[22],
    'SibSp':[1],
    'Parch':[0],
    'Fare':[7.25],
    'Embarked':[2]
})

prediction = model.predict(new_passenger)

if prediction[0] == 1:
    print("Prediction : Passenger Survived")
else:
    print("Prediction : Passenger Did Not Survive")

# ==========================================================
# STEP 12 : Save Model
# ==========================================================

pickle.dump(model, open("titanic_model.pkl", "wb"))

print("\nModel Saved Successfully as titanic_model.pkl")

# ==========================================================
# STEP 13 : Feature Importance
# ==========================================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

importance = importance.sort_values(by='Coefficient', ascending=False)

print("\n")
print("=" * 60)
print("Feature Importance")
print("=" * 60)
print(importance)

plt.figure(figsize=(8,5))

sns.barplot(
    data=importance,
    x='Coefficient',
    y='Feature'
)

plt.title("Feature Importance")

plt.show()

print("\n")
print("=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)