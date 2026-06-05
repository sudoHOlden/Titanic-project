import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('Titanic-Dataset.csv')
df = df.drop(['Cabin','PassengerId','Ticket','Name'], axis=1)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna('S')
# df['Family_size'] = df['SibSp'] + df['Parch'] + 1
df['Sex'] = df.apply(lambda x: 1 if x['Sex'] == 'female' else 0, axis=1)
df = pd.get_dummies(df, 'Embarked',dtype=int)
y = df['Survived']
X = df.drop('Survived', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(df.head(10).to_string())
print(df.info())
print(accuracy_score(y_test, y_pred))
features = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
print(features.sort_values(by='importance', ascending=False))
print(model.feature_importances_)