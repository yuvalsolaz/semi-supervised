# imports for array-handling and plotting
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# keep our keras backend tensorflow quiet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
# for testing on CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

# keras imports for the dataset and building our neural network
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import collections
#PLabel = collections.namedtuple('Label', ['class_label', 'probability'], verbose=True)

def run():
    # load mnist train & test
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # print the shape before we reshape and normalize
    print("X_train shape", X_train.shape)
    print("y_train shape", y_train.shape)
    print("X_test shape", X_test.shape)
    print("y_test shape", y_test.shape)

    # building the input vector from the 28x28 pixels
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalizing the data to help with the training
    X_train /= 255
    X_test /= 255

    # print the final input shape ready for training
    print("Train matrix shape", X_train.shape)
    print("Test matrix shape", X_test.shape)

    # one-hot encoding using keras' numpy-related utilities
    n_classes = 10
    print("Shape before one-hot encoding: ", y_train.shape)
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)
    print("Shape after one-hot encoding: ", Y_train.shape)

    # split train to labeled & unlabeld
    labeled_size = .10
    X_labeled, X_unlabeled, Y_labeled, Y_unlabeled = train_test_split(X_train, Y_train,
                                                                      train_size=labeled_size, stratify=y_train)
    print("X_labeled shape", X_labeled.shape)
    print("Y_labeled shape", Y_labeled.shape)
    print("X_unlabeled shape", X_unlabeled.shape)
    print("Y_unlabeled shape", Y_unlabeled.shape)

    # build model :
    model = build_model()

    # training the model on the labeled data and saving metrics in history
    history = model.fit(X_labeled, Y_labeled,
                        batch_size=128, epochs=4,
                        verbose=2,
                        validation_data=(X_test, Y_test))

    # check accuracy on test data
    metrics = model.evaluate(X_test, y=Y_test, batch_size=128)
    for i in range(len(model.metrics_names)):
        print(str(model.metrics_names[i]) + ": " + str(metrics[i]))

    # use the model to predict unlabeled data and create pseudo labeles
    threshold = .999
    predict = np.argmax(model.predict_proba(X_unlabeled),axis=1)
    # select unlabeled samples with predictions probabilty above threshold :
    pseudo_X_train = X_unlabeled[predict > threshold]
    pseudo_Y_train = Y_unlabeled[predict > threshold]

    print ('using threshold of {} - predicts {} pseudo labels from {} unlabeld samples'. \
           format(threshold, len(pseudo_X_train), len(X_unlabeled)))



def build_model():
    # building a linear stack of layers with the sequential model
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    print(model.summary())

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model



def plot_digits(X_train, y_train):
    fig = plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[i]))
    plt.xticks([])
    plt.yticks([])
    fig


def plot_results(model):
    # plotting the metrics
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(model.history['acc'])
    plt.plot(model.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')

    plt.subplot(2,1,2)
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.tight_layout()
    plt.show()
    fig


if __name__ == '__main__':
    run()