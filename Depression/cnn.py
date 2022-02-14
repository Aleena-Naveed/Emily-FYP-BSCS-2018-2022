from __future__ import print_function
# import boto
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from plot_metrics import plot_accuracy, plot_loss, plot_roc_curve
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from tensorflow.keras import regularizers

# access_key = os.environ['AWS_ACCESS_KEY_ID'] r
# access_secret_key = os.environ['AWS_SECRET_ACCESS_KEY'] r
np.random.seed(15)  # for reproducibility


"""
CNN used to classify spectrograms of normal participants (0) or depressed
participants (1). Using Theano backend and Theano image_dim_ordering:
(# channels, # images, # rows, # cols)
(1, 3040, 513, 125)
"""
def preprocess(X_train, X_test):
    """
    Convert from float64 to float32 and normalize normalize to decibels
    relative to full scale (dBFS) for the 4 sec clip.
    """
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_train])
    X_test = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_test])
    return X_train, X_test


def prep_train_test(X_train, y_train, X_test, y_test, nb_classes):
    """
    Prep samples ands labels for Keras input by noramalzing and converting
    labels to a categorical representation.
    """
    print('Train on {} samples, validate on {}'.format(X_train.shape[0],
                                                       X_test.shape[0]))

    # normalize to dBfS
    X_train, X_test = preprocess(X_train, X_test)

    # Convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, X_test, Y_train, Y_test


def keras_img_prep(X_train, X_test, img_dep, img_rows, img_cols):
    """
    Reshape feature matrices for Keras' expexcted input dimensions.
    For 'th' (Theano) dim_order, the model expects dimensions:
    (# channels, # images, # rows, # cols).
    """
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    return X_train, X_test, input_shape


def cnn(X_train, y_train, X_test, y_test, batch_size,
        nb_classes, epochs, input_shape):
    """
    The Convolutional Neural Net architecture for classifying the audio clips
    as normal (0) or depressed (1).
    """
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='valid', strides=1,
                     input_shape=input_shape, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)
                     ))

    model.add(MaxPooling2D(pool_size=(4, 3), strides=(1, 3)))

    # model.add(Conv2D(32, (1, 3), padding='valid', strides=1,
    #                  input_shape=input_shape, activation='relu'))
    model.add(Conv2D(32, (1, 3), padding='valid', strides=1,
                     activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))

    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))

    model.add(Flatten())
    model.add(Dense(512,
                    activation='relu',
                    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=regularizers.l2(1e-4),
                    activity_regularizer=regularizers.l2(1e-5)
                    ))
    model.add(Dense(512,
                    activation='relu',
                    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=regularizers.l2(1e-4),
                    activity_regularizer=regularizers.l2(1e-5)
                    ))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                        verbose=1, validation_data=(X_test, y_test))

    # Evaluate accuracy on test and train sets
    score_train = model.evaluate(X_train, y_train, verbose=0)
    print('Train accuracy:', score_train[1])
    score_test = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', score_test[1])

    return model, history


def model_performance(model, X_train, X_test, y_train, y_test):
    """
    Evaluation metrics for network performance.
    """
    y_test_pred = model.predict_classes(X_test)
    y_train_pred = model.predict_classes(X_train)

    y_test_pred_proba = model.predict_proba(X_test)
    y_train_pred_proba = model.predict_proba(X_train)

    # Converting y_test back to 1-D array for confusion matrix computation
    y_test_1d = y_test[:, 1]

    # Computing confusion matrix for test dataset
    conf_matrix = standard_confusion_matrix(y_test_1d, y_test_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    return y_train_pred, y_test_pred, y_train_pred_proba, \
        y_test_pred_proba, conf_matrix


def standard_confusion_matrix(y_test, y_test_pred):
    """
    Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D

    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_test, y_test_pred)
    return np.array([[tp, fp], [fn, tn]])


if __name__ == '__main__':
    model_id = input("Enter model id: ")

    # Load from S3 bucket
    print('Retrieving from S3...')
    X_train = np.load(
        r'../data/processed/train_samples.npz')
    y_train = np.load(
        r'../data/processed/train_labels.npz')
    X_test = np.load(
        r'../data/processed/test_samples.npz')
    y_test = np.load(
        r'../data/processed/test_labels.npz')

    X_train, y_train, X_test, y_test = \
        X_train['arr_0'], y_train['arr_0'], X_test['arr_0'], y_test['arr_0']

    # CNN parameters
    batch_size = 64
    nb_classes = 2
    epochs = 50

    # normalalize data and prep for Keras
    print('Processing images for Keras...')
    X_train, X_test, y_train, y_test = prep_train_test(X_train, y_train,
                                                       X_test, y_test,
                                                       nb_classes=nb_classes)

    # 513x125x1 for spectrogram with crop size of 125 pixels
    img_rows, img_cols, img_depth = X_train.shape[1], X_train.shape[2], 1

    # reshape image input for Keras
    # used Theano dim_ordering (th), (# chans, # images, # rows, # cols)
    X_train, X_test, input_shape = keras_img_prep(X_train, X_test, img_depth,
                                                  img_rows, img_cols)

    # run CNN
    print('Fitting model...')
    model, history = cnn(X_train, y_train, X_test, y_test, batch_size,
                         nb_classes, epochs, input_shape)

    # evaluate model
    print('Evaluating model...')
    y_train_pred, y_test_pred, y_train_pred_proba, y_test_pred_proba, \
        conf_matrix = model_performance(
            model, X_train, X_test, y_train, y_test)

    accuracy = float(conf_matrix[0][0] +
                     conf_matrix[1][1]) / np.sum(conf_matrix)

    # save model to locally
    print('Saving model locally...')
    model_name = './cnn_{model_id} {acc}.h5'.format(
        model_id=model_id, acc=accuracy)
    model.save(model_name)

    # custom evaluation metrics
    print('Calculating additional test metrics...')
    accuracy = float(conf_matrix[0][0] +
                     conf_matrix[1][1]) / np.sum(conf_matrix)
    precision = float(conf_matrix[0][0]) / \
        (conf_matrix[0][0] + conf_matrix[0][1])
    recall = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[1][0])
    f1_score = 2 * (precision * recall) / (precision + recall)
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1-Score: {}".format(f1_score))

    # plot train/test loss and accuracy. saves files in working dir
    print('Saving plots...')
    plot_loss(history, model_id)
    plot_accuracy(history, model_id)
    plot_roc_curve(y_test[:, 1], y_test_pred_proba[:, 1], model_id)

    # save model S3
