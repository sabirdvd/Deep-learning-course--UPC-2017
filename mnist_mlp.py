
from __future__ import print_function

import keras
from keras.datasets import mnist

# Define the NN architecture
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

# plot
import matplotlib
import matplotlib.pyplot as plt

# print the version
#print 'using keras version', keras.__version__


# Confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# Saving  model and weight
from keras.models import model_from_json

# Parameter
batch_size = 128
num_classes = 10
epochs = 2

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Adapt the data as an input of fullly connected (flatten to 1D)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Normalize data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# State-of-the-art model
#model = Sequential()
#model.add(Dense(512, activation='relu', input_shape=(784,)))
#model.add(Dropout(0.2))
#model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(10, activation='softmax'))

#our model
# Two hidden Layer
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(784,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))



model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])



history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
# Score
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Store plot
matplotlib.use('Agg')

# Accuracy plot
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc= 'upper left')
plt.savefig('model_accuracy.pdf')
plt.close()

#  loss plot

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc= 'upper left')
plt.savefig('model_loss.pdf')

# confusion matrix
# compute the probabilities
##Y_pred = nn.predict(x_test)
Y_pred = model.predict(x_test)

#Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)

# plot statistics
###print ('Analysis of result')
###target_name = ['0','1','2','3','4','5','6','7','8','9']
###print(classification_report(np.argmax(y_test, axis=1), y_pred, target_name=target_name))
###print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))

# Saving  model and weight
nn_json = model.to_json()

with open ('nn.json', 'w') as json_file:
            json_file.write(nn_json)
#weight_file = "weight-MNIST_" + "str(score[1])+".hdf5
#model.save_weights(weights_file, overwrite=True)
model.save_weights('my_model_weights.h5')
        
# load the the model and weights
#json_file = open('nn.json', 'r')
#nn_json = json_file.read()
#json_file.close()
#nn = model_from_json(nn_json)
#nn.load_weights(weight_file)
#model.load_weights(weight_file)

model.load_weights('my_model_weights.h5')


           
           
