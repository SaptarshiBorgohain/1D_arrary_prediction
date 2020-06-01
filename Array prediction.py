#Do it in jupyter notebook
#In 1st cell
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk

#2nd cell
x = [0, 1, 0],[1, 0, 0],[1, 1, 1],[1, 1, 0],
y = [1, -1, 1, -1]

#3rd cell
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.20,random_state=0)

#4th cell
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
classifier.fit(x_train, y_train)

#5th cell
y_pred = classifier.predict(x_test)
y_pred

#6th cell
y_test

#7th cell
new_prediction = classifier.predict(np.array([[0, 1, 1],[1, 0, 1]]))
new_prediction