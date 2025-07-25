import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import joblib
# Load dataset
df = pd.read_csv("tested.csv")
df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Cabin', 'Ticket'], axis=1, inplace=True)
# Fill missing numeric values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
# Fill missing categorical values
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# Encode categorical features
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])
# Final check for missing values
print("Any missing values?\n", df.isnull().sum())
print(df.head())
# Splitting data
X_data = df.drop('Survived', axis=1)
y = df['Survived']  # target
# Training data
X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, random_state=42)
# Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
ra_score = model.score(X_test, y_test)
print(f"Random Forest Accuracy: {ra_score:.2f}")
joblib.dump(model, "titanic_model.pkl")
model = joblib.load("titanic_model.pkl")
# AdaBoost
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_model.fit(X_train, y_train)
ada_score = ada_model.score(X_test, y_test)
print(f"AdaBoost Accuracy: {ada_score:.2f}")
# XGBoost
xg_model = xg.XGBClassifier(eval_metric='logloss', n_estimators=100, random_state=42)
xg_model.fit(X_train, y_train)
xg_score = xg_model.score(X_test, y_test)
print(f"XGBoost Accuracy: {xg_score:.2f}")
# Visualizing content
importances = model.feature_importances_
feat_names = X_data.columns
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feat_names, color='steelblue')
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
