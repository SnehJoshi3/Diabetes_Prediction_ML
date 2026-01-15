import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data = pd.read_csv('/kaggle/input/diabetes-csv/diabetes.csv')
data.head()

data.info()

plt.figure(figsize = (12,6))
sns.countplot(x = 'Outcome' , data = data)
plt.show()

plt.figure(figsize =(12,12))
for i ,col in enumerate(['Pregnancies','Glucose', 'BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']):
    plt.subplot(3,3,i+1)
    sns.boxplot(x = col , data = data)
plt.show()

sns.pairplot(hue ='Outcome', data = data)
plt.show()

plt.figure(figsize = (12,12))
for i , col in enumerate(['Pregnancies','Glucose', 'BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']):
    plt.subplot(3,3,i+1)
    sns.histplot(x = col , data = data , kde = True)
plt.show()

scx = StandardScaler()
X_scaled = scx.fit_transform(data.drop(columns=['Outcome']))
X = pd.DataFrame(X_scaled, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

X.head()

y = data['Outcome']
y.head()

X_train , X_test, y_train  , y_test = train_test_split(X,y ,test_size = 0.3 ,random_state = 42)

test_scores =[]
train_scores = []

for i in range(1,15):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train , y_train)
    train_scores.append(knn.score(X_train , y_train))
    test_scores.append(knn.score(X_test , y_test))

max_train_score = max(train_scores)
train_score_index = [i for i , v in enumerate(train_scores) if v == max_train_score]

print(max_train_score)

Knn = KNeighborsClassifier(13)
Knn.fit(X_train , y_train)
Knn.score(X_test,y_test)

y_pred = knn.predict(X_test)
confusion_matrix(y_test , y_pred)

print(classification_report(y_test,y_pred))
