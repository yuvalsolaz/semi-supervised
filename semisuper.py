# imports for array-handling and plotting
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
    labeled_size = .20
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


    # use the model to predict unlabeled data and create pseudo labeles
    predict = model.predict_proba(X_unlabeled)

    # reverse one hot encoding to classes
    class_proba = [(max(x),np.argmax(x)) for x in predict] #.sort(reverse=True)
    class_proba.sort(reverse=True)
    threshold = .999
    pseudo_labels = [p for p in class_proba[0:] if p[0] > threshold ]
    print ('using threshold of {} - predicts {} pseudo labels from {} unlabeld samples'. \
           format(threshold, len(pseudo_labels), len(X_unlabeled)))

    #plot_results(history)


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


def create_augmented_train(X, y):
      '''
      Create and return the augmented_train set that consists
      of pseudo-labeled and labeled data.
      '''
      num_of_samples = int(len(self.unlabled_data) * self.sample_rate)

      # Train the model and creat the pseudo-labels
      self.model.fit(X, y)
      pseudo_labels = self.model.predict(self.unlabled_data[self.features])

      # Add the pseudo-labels to the test set
      pseudo_data = self.unlabled_data.copy(deep=True)
      pseudo_data[self.target] = pseudo_labels

      # Take a subset of the test set with pseudo-labels and append in onto
      # the training set
      sampled_pseudo_data = pseudo_data.sample(n=num_of_samples)
      temp_train = pd.concat([X, y], axis=1)
      augemented_train = pd.concat([sampled_pseudo_data, temp_train])

      return shuffle(augemented_train)

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