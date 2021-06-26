import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from joblib import load,dump
from sklearn.neural_network import MLPClassifier

hyperparams = {
        "SVM" : {
            "kernel" : ["rbf","linear","poly","sigmoid"],
            "C" : [0.1,1,10,100,1000],
            "gamma" : [0.001, 0.0001, 0.00001]
        },
        "Boosting" : {
            "n_estimators" : list(range(1,100,5)),
            "max_features" : ["sqrt","log2",None],
            "learning_rate" : [0.1, 0.01, 0.001]
        },
        "KNN":{
            "n_neighbors" : list(range(2,50)),
            "weights" : ["uniform", "distance"]
        },
        #"NeuralNetwork":{
        #    "optimizer" : ['SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
        #    "weight_constraint" : [3, 4, 5],
        #    "dropout_rate" : [0.0, 0.2, 0.4, 0.5, 0.8]
        #}
        "NeuralNetwork":{
            "activation" : ['identity', 'logistic', 'tanh', 'relu'],
            "solver" : ['lbfgs', 'sgd', 'adam'],
            "learning_rate" : ['constant', 'invscaling', 'adaptive']
        }
}


data = pd.read_csv("final_data.csv")


x_train, x_test, y_train, y_test = train_test_split(data.drop("label",axis=1),data["label"],test_size=0.3)
"""
# SVM
model1 = SVC()
grid1 = GridSearchCV(estimator=model1,param_grid=hyperparams["SVM"],cv=10)
grid1.fit(x_train,y_train)

model1 = grid1.best_estimator_

predictions1 = model1.predict(x_test)
print(classification_report(y_test,predictions1))
dump(model1,"modelSVM.joblib")


# Boosting
model2 = GradientBoostingClassifier()
grid2 = GridSearchCV(estimator=model2,param_grid=hyperparams["Boosting"],cv=10)
grid2.fit(x_train,y_train)

model2 = grid2.best_estimator_

predictions2 = model2.predict(x_test)
print(classification_report(y_test,predictions2))
dump(model2,"modelBoosting.joblib")

# KNN
model3 = KNeighborsClassifier()
grid3 = GridSearchCV(estimator=model3,param_grid=hyperparams["KNN"],cv=10)
grid3.fit(x_train,y_train)

model3 = grid3.best_estimator_

predictions3 = model3.predict(x_test)
print(classification_report(y_test,predictions3))
dump(model3,"modelKNN.joblib")

#Neural Network
def create_model(optimizer='Adam',dropout_rate=0.0, weight_constraint=0):
    print("Optimizer : ",optimizer,",Dropout : ",dropout_rate,"Weight Constraint : ",weight_constraint)
    model = Sequential()
    model.add(Dense(12, input_dim=len(x_train.columns), activation='relu',kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=optimizer)
    return model

model4 = KerasClassifier(build_fn=create_model ,batch_size=32,epochs=100, verbose=1)
grid4 = GridSearchCV(estimator=model4, param_grid=hyperparams["NeuralNetwork"], n_jobs=-1, cv=3)
grid4.fit(x_train, y_train,validation_split=0.3)
model4 = grid4.best_estimator_
predictions4=model4.predict(x_test)
print(classification_report(y_test,predictions4))
dump(model4, "modelNN.joblib")
"""

model4 = MLPClassifier(hidden_layer_sizes=(20,20),max_iter=500)
grid4 = GridSearchCV(estimator=model4,param_grid=hyperparams["NeuralNetwork"],cv=10,verbose=1)
grid4.fit(x_train,y_train)

model4 = grid4.best_estimator_

predictions4 = model4.predict(x_test)
print(classification_report(y_test,predictions4))
dump(model4,"model2NN.joblib")
