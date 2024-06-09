#Importing Libraries
import pandas as pd
 
#Importing the dataset
dataset = pd.read_csv('C:\\Users\\LENOVO\\Desktop\\projects\\breast cancer using logistic\\breast_cancer.csv')
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values
print(X)
print(y)

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X,y, test_size=0.2, random_state=0)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

#Training the Logistic Regression Model on the Training set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#Predicting the Test Set results
y_pred=classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)

#Computing the accuracy with k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f}%".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f}%".format(accuracies.std()*100))