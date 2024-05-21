import time
import tensorflow as tf
from tensorflow import keras
from scipy.fft import fft
import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import plot_confusion_matrix
import re
from tensorflow import keras
# from tensorflow.keras.utils import to_categorical
# from keras.utils import to_categorical
from Transformer import ImageTransformer, LogScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import Activation, Dense, MaxPooling2D, Flatten
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.svm import SVC
from tqdm import tqdm
from scipy.linalg import block_diag
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt  # matplotlib.pyplot is used for plotting a figure in the same way with matlab and making some change on it.

plt.rcParams["font.family"] = "Times New Roman"  # rcpParams Tips for customizing the properties and default styles of Matplotlib.

import warnings  # ''ignore'' means not important warning to stop the program. "never print matching warnings"

warnings.filterwarnings("ignore")

from all_args import run_shell_options
from utils.utils import Logger
import sys

start_time = time.time()


def b_fft(x, it):  # fast fourier transform
    b = fft(x)  # If X is a matrix, then fft(X) treats the columns of X as vectors and returns the Fourier transform of each column.
    b_real = b.real
    b_real = np.where(b_real < 0, b_real, 0)

    b_imag = b.imag
    b_imag = np.where(b_imag < 0, b_imag, 0)

    scale_fft = LogScaler()  # Log normalize and scale data
    b_real_norm = scale_fft.fit_transform(np.abs(b_real + 1))
    b_imag_norm = scale_fft.fit_transform(np.abs(b_imag + 1))

    x_fft_real = it.transform(b_real_norm);
    x_fft_imag = it.transform(b_imag_norm);

    return x_fft_real, x_fft_imag

def classifiers(x_train, x_test, train_label, test_label):
    # SVM
    svm_clf = svm.SVC()
    svm_clf.fit(x_train, train_label)
    svm_pred_label = svm_clf.predict(x_test)
    svm_acc = accuracy_score(test_label, svm_pred_label)
    svm_f1 = f1_score(test_label, svm_pred_label, average='micro')

    # RF
    rf_clf = RandomForestClassifier(max_depth=2, random_state=0)
    rf_clf.fit(x_train, train_label)
    rf_pred_label = rf_clf.predict(x_test)
    rf_acc = accuracy_score(test_label, rf_pred_label)
    rf_f1 = f1_score(test_label, rf_pred_label, average='micro')

    #KNN
    knn_cls = KNeighborsClassifier(n_neighbors=3)
    knn_cls.fit(x_train, train_label)
    knn_pred_label = knn_cls.predict(x_test)
    knn_acc = accuracy_score(test_label, knn_pred_label)
    knn_f1 = f1_score(test_label, knn_pred_label, average='micro')

    return svm_acc, svm_f1, rf_acc, rf_f1, knn_acc, knn_f1

def main_func():
    args = run_shell_options()
    fileName = args.run_dataset

    # save Terminal outputs
    out_dir = args.out_dir + '/' + args.run_dataset.split('.')[0] + '/cpm/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log_file_name = out_dir+'Terminal_'+str(start_time)+'.log'
    # record normal print messages
    sys.stdout = Logger(log_file_name)
    # record traceback error messages
    sys.stderr = Logger(log_file_name)


    # Data clean
    filepath = 'Datasets/RowCount/new/' + fileName if '-RawCount-new' in fileName else 'Datasets/RowCount/' + fileName
    Dataset = pd.read_csv(filepath, index_col=0, header=None, dtype='unicode').T
    Dataset = Dataset.rename(columns={Dataset.columns[0]: 'Unnamed: 0'})
    Dataset.drop(columns=['Unnamed: 0'], inplace=True)  # omitting the first column consist of feature names
    if 'NA' in Dataset.columns: Dataset.drop(columns=['NA'], inplace=True)
    Dataset.dropna(inplace=True)  # Drop rows with missing indices or values.(drop all missing data)

    labels = np.array(Dataset.columns.tolist())  # Get list from pandas dataframe column or row
    for i in range(len(labels)):
        labels[i] = ''.join(labels[i].split('.'))

    data = Dataset.T.values  # Pandas DataFrame.values attribute return a Numpy representation of the given DataFrame.
    data = data.astype('int64')

    print('len of unique classes: ' + str(len(list(set(labels)))))
    print('Input Type: ' + str(type(data[0][0])))

    #CPM normalization
    data = np.nan_to_num(data)
    sum_data = np.sum(data, axis=1)
    sum_data = sum_data.reshape(sum_data.shape[0], -1)
    cpm_data = np.divide(data, sum_data)
    cpm_data = cpm_data * 10**6

    # Train Test split
    classes, indices = np.unique(labels, return_inverse=True)
    X_Train, X_Test, y_train, y_test, train_label, test_label = train_test_split(
        cpm_data, labels, indices, test_size=0.2, random_state=23, stratify=labels)


    svm_acc, svm_f1, rf_acc, rf_f1, knn_acc, knn_f1 = classifiers(X_Train, X_Test, train_label, test_label)
    print("SVM Acc: {}, SVM F1: {}, RF Acc: {}, RF F1: {}, KNN Acc: {}, KNN F1: {}"
        .format(svm_acc, svm_f1, rf_acc, rf_f1, knn_acc, knn_f1))

    ln = LogScaler()  # Log normalize and scale data
    X_train = ln.fit_transform(X_Train)
    X_test = ln.transform(X_Test)

    # prepare frames
    train_frames = []
    for i in range(len(classes)):
        train_frame = X_train[np.where(train_label == i)[0], :]
        train_frames.append(train_frame)

    # Apply Transformer and get fitters
    fitters = []
    for i in range(len(classes)):
        it = ImageTransformer(feature_extractor='tsne',
                            pixels=100, random_state=12345,
                            n_jobs=-1)

        train_frame = np.array(train_frames[i])

        _ = it.fit(train_frame, plot=False)

        fitters.append(it)

    # Prepare training images
    Pixel = 100
    x_train_img_fft = np.zeros((len(X_train), len(classes) * Pixel + 1, len(classes) * Pixel, 3))
    for j in range(len(X_train)):
        train_frame = X_train[j]
        class_data = []
        block_img_0 = []
        block_img_1 = []
        block_img_2 = []
        for it in fitters:
            x_train_img = it.transform(X_train[j])
            block_img_0 = block_diag(block_img_0, x_train_img[0, :, :, 0])

            x_train_fft_real, x_train_fft_imag = b_fft(train_frame, it)
            block_img_1 = block_diag(block_img_1, x_train_fft_real[0, :, :, 0])
            block_img_2 = block_diag(block_img_2, x_train_fft_imag[0, :, :, 0])

        x_train_img_fft[j, :, :, 0] = block_img_0
        x_train_img_fft[j, :, :, 1] = block_img_1
        x_train_img_fft[j, :, :, 2] = block_img_2

    x_train_img_fft[x_train_img_fft > 1] = 1
    X_train_imgs = x_train_img_fft

    y_predict_label = []
    x_test_img_fft = np.zeros((len(X_test), len(classes) * Pixel + 1, len(classes) * Pixel, 3))
    for j in range(len(X_test)):
        test_frame = X_test[j]
        class_data = []
        block_img_0 = []
        block_img_1 = []
        block_img_2 = []
        for it in fitters:
            x_test_img = it.transform(X_test[j])
            block_img_0 = block_diag(block_img_0, x_test_img[0, :, :, 0])

            x_test_fft_real, x_test_fft_imag = b_fft(test_frame, it)
            block_img_1 = block_diag(block_img_1, x_test_fft_real[0, :, :, 0])
            block_img_2 = block_diag(block_img_2, x_test_fft_imag[0, :, :, 0])

        x_test_img_fft[j, :, :, 0] = block_img_0
        x_test_img_fft[j, :, :, 1] = block_img_1
        x_test_img_fft[j, :, :, 2] = block_img_2

    x_test_img_fft[x_test_img_fft > 1] = 1
    X_test_imgs = x_test_img_fft

    print('Shape of X_train_imgs: ' + str(X_train_imgs.shape))
    print('Shape of X_test_imgs: ' + str(X_test_imgs.shape))

    image_time = time.time()
    imag_run_time = image_time - start_time
    print(' image_run_time: ' + str(imag_run_time))

    X_train_imgs_cp = X_train_imgs
    X_test_imgs_cp = X_test_imgs
    import cv2

    X_train_imgs_resize = []
    X_test_imgs_resize = []

    # INTER_NEAREST - a nearest-neighbor interpolation
    # INTER_LINEAR - a bilinear interpolation (used by default)
    # INTER_AREA - resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
    # INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
    # INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood

    for i in range(len(X_train_imgs_cp)):
        X_train_imgs_resize.append(cv2.resize(X_train_imgs_cp[i], dsize=(200, 200), interpolation=cv2.INTER_LANCZOS4))

    for i in range(len(X_test_imgs_cp)):
        X_test_imgs_resize.append(cv2.resize(X_test_imgs_cp[i], dsize=(200, 200), interpolation=cv2.INTER_LANCZOS4))

    X_train_imgs_resize = np.array(X_train_imgs_resize)
    X_test_imgs_resize = np.array(X_test_imgs_resize)

    print('Shape of X_train_imgs_resize: ' + str(X_train_imgs_resize.shape))
    print('Shape of X_test_imgs_resize: ' + str(X_test_imgs_resize.shape))

    resiz_time = time.time()
    resize_time = resiz_time - image_time
    print(' imgresize_run_time: ' + str(resize_time))

    # CNN model
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=X_train_imgs_resize[0].shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(len(classes)))
    model.add(Activation('sigmoid'))
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='auto')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=1, min_delta=1e-4, mode='auto')
    # model.summary()

    model.fit(X_train_imgs_resize, train_label, batch_size=32, epochs=90, verbose=1,
            callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
            validation_split=0.2)

    # Evaluate
    y_pred_0 = model.predict(X_test_imgs_resize)
    y_predict_label = np.argmax(y_pred_0, axis=1)


    acc = accuracy_score(y_predict_label, test_label)
    f1_mac = f1_score(y_predict_label, test_label, average='macro')
    f1_mic = f1_score(y_predict_label, test_label, average='macro')
    f1_wght = f1_score(y_predict_label, test_label, average='macro')
    sensitivity_mac = recall_score(y_predict_label, test_label, average='macro')
    sensitivity_mic = recall_score(y_predict_label, test_label, average='micro')
    sensitivity_wght = recall_score(y_predict_label, test_label, average='micro')

    tn, fp, fn, tp = confusion_matrix(y_predict_label, test_label, labels=[0, 1]).ravel()
    specificity = tp/(tn+fp)



    end_time = time.time()
    running_time = end_time - start_time
    cnn_time = end_time - resiz_time

    print(fileName + ' Accuracy: ' + str(acc))
    print(fileName + ' specificity: ' + str(specificity))
    print(' run_time: ' + str(running_time))
    print(' cnn_run_time: ' + str(cnn_time))

    print("f1_mac: {}, f1_mic: {}, f1_wght: {}".format(f1_mac, f1_mic, f1_wght))
    print("sensitivity_mac: {}, sensitivity_mic: {}, sensitivity_wght: {}".format(sensitivity_mac, sensitivity_mic, sensitivity_wght))


if __name__ == "__main__":
    main_func()