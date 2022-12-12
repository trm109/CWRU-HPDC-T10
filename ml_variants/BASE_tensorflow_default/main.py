#imports
import pandas as pd
pd.__version__
from sklearn.model_selection import train_test_split


def explain(val, name):
    print(name + "\n")
    print(type(val))
    print(val)
    return

#
dataframe = pd.read_csv('./Churn.csv')
#x is the dummies of the Churn.csv, without the columns 'Churn' or 'Customer ID', or the first row?
x = pd.get_dummies(dataframe.drop(['Churn','Customer ID'], axis=1))

explain(x,'x') # Type DataFrame, 7044 x 6575 :  full set sans Churn/Customer ID columns.

#y is the dataframe, but with Churn's values converted from strings to 1/0s
y = dataframe['Churn'].apply(lambda x:1 if x=='Yes' else 0)

explain(y,'y') # Type Series, 1x7044 : the true/false booleanization of the Churn column only.

#Splits columns,
#x_train : 80% of the x data, used to train.
#x_test : 20% of the x data, used to test
#y_train : 80% of the y column, used to train (same items as x)
#y_test : 20% of the y data, used to test.
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2)

#explain(x_train,'x_train') # type DataFrame, 5635x6575  : larger training set?
#explain(x_test, 'x_test')  # type DataFrame, 1409x6575  : smaller test set?

#explain(y_train,'y_train')
#explain(y_test,'y_test')

#TensorFlow imports.
from tensorflow.keras.models import Sequential, load_model #Models
from tensorflow.keras.layers import Dense #Layers
from sklearn.metrics import accuracy_score #diagnostics.

#Building the model
model = Sequential() #Create a new sequential model.

#Adding the 'sequential' steps of the machine learning model. the layers(?)
#first layer, a Dense, using Relu (linear) w/ unit size 32, with the input dimension of all the 'categories' in x
model.add(Dense(units=32, activation='relu', input_dim=len(x_train.columns)))
#Second layer, a second Dense, also using relu, but with a larger unit size of 64
model.add(Dense(units=64, activation='relu'))
#Third layer, final layer using the more expenseive 'sigmoid'
model.add(Dense(units=1, activation='sigmoid'))
#compile the model into something runnable
#loss function of Binary Cross entropy
#Optimizer function of sgd
#Metrics of 'accuracy' (what gets evaluated by the model.)
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy',jit_compile=True)


#Actually start the training.
model.fit(x_train, y_train, epochs=200, batch_size=32)

#Test it!
#Create a new predication
y_hat = model.predict(x_test)
explain(y_hat, 'y_hat') # type ndarray, 
#Binarize every value in y_hat to get a yes or no
y_hat = [0 if val < 0.5 else 1 for val in y_hat]
#Test
accuracy = accuracy_score(y_test, y_hat) #Prints the accuracy of the model, give the test set.
print("Accuracy: " + str(accuracy))
#Save the model.
model.save('tfmodel')

print(f"Completed running!")