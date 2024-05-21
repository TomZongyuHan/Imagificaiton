import os
import cv2
import sys
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import Activation, Dense, MaxPooling2D, Flatten
from scipy.linalg import block_diag
from sklearn.metrics import accuracy_score, f1_score, recall_score
import matplotlib.pyplot as plt  # matplotlib.pyplot is used for plotting a figure in the same way with matlab and making some change on it.
import warnings  # ''ignore'' means not important warning to stop the program. "never print matching warnings"

from utils.utils import Logger
from pyscripts.b_fft import b_fft
from pyscripts.all_args import run_shell_options
from pyscripts.TroditionalClassifiers import classifiers
from pyscripts.Transformer import ImageTransformer, LogScaler

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Times New Roman"  # rcpParams Tips for customizing the properties and default styles of Matplotlib.

def main_func(args):
    start_time = time.time()

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

    """
    #################################### Stage 1 #####################################
    Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc.
    """
    print("#################### Stage 1: Data preparation #####################")
    
    # Refer the data file path
    data_root_dir = args.data_dir # default Datasets/
    filepath = data_root_dir+'RowCount/new/'+fileName if '-RawCount-new' in fileName else data_root_dir+'RowCount/' + fileName
    
    # Read data
    Dataset = pd.read_csv(filepath, index_col=0, header=None, dtype='unicode').T
    
    # Data clean
    Dataset = Dataset.rename(columns={Dataset.columns[0]: 'Unnamed: 0'})
    Dataset.drop(columns=['Unnamed: 0'], inplace=True)  # omitting the first column consist of feature names
    if 'NA' in Dataset.columns: Dataset.drop(columns=['NA'], inplace=True)
    Dataset.dropna(inplace=True)  # Drop rows with missing indices or values.(drop all missing data)

    labels = np.array(Dataset.columns.tolist())  # Get list from pandas dataframe column or row
    for i in range(len(labels)):
        labels[i] = ''.join(labels[i].split('.'))

    data = Dataset.T.values  # Pandas DataFrame.values attribute return a Numpy representation of the given DataFrame.
    try:
        data = data.astype('int64')
    except:
        data = data.astype('float64')

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

    print("Shape of X_train:{}".format(X_train.shape))
    print("Shape of X_test:{}".format(X_test.shape))
    print("Shape of y_train:{}".format(y_train.shape))
    print("Shape of y_test:{}".format(y_test.shape))

    # get results of traditional classification methods
    svm_acc, svm_f1, rf_acc, rf_f1, knn_acc, knn_f1 = classifiers(X_Train, X_Test, train_label, test_label)
    print("SVM Acc: {}, SVM F1: {}, RF Acc: {}, RF F1: {}, KNN Acc: {}, KNN F1: {}"
        .format(svm_acc, svm_f1, rf_acc, rf_f1, knn_acc, knn_f1))

    # Log normalize and scale data
    ln = LogScaler()
    X_train = ln.fit_transform(X_Train)
    X_test = ln.transform(X_Test)

    # prepare frames
    train_frames = []
    for i in range(len(classes)):
        train_frame = X_train[np.where(train_label == i)[0], :]
        train_frames.append(train_frame)

    '''
    #################################### Stage 2 #####################################
    Stage 2: Apply Transformer and get fitters
    '''
    print("#################### Stage 2: Apply Transformer #####################")

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

    # resize
    X_train_imgs_resize = []
    X_test_imgs_resize = []
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

    '''
    #################################### Stage 3 #####################################
    Stage 3: Train CNN Model
    '''
    print("#################### Stage 3: Training #####################")
    
    # get arguments
    monitor = args.monitor
    es_patience = args.es_patience
    verbose = args.verbose
    mode = args.mode
    save_best_only = args.save_best_only
    factor = args.factor
    lr_patience = args.lr_patience
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    validation_split = args.validation_split
    in_shape = X_train_imgs[0].shape

    # Build CNN model
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

    earlyStopping = EarlyStopping(monitor=monitor, patience=es_patience, verbose=verbose, mode=mode)
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=save_best_only, monitor=monitor, mode=mode)
    reduce_lr_loss = ReduceLROnPlateau(monitor=monitor, factor=factor, patience=lr_patience, verbose=verbose, min_delta=lr, mode=mode)
    # model.summary()

    # start training
    model.fit(X_train_imgs_resize, train_label, batch_size=batch_size, epochs=epochs, verbose=verbose,
            callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
            validation_split=validation_split)

    '''
    #################################### Stage 4 #####################################
    Stage 4: Evaluation
    '''
    print("#################### Stage 4: Evaluation #####################")
    
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

    print(fileName + ' CNN Acc: ' + str(acc))
    print(fileName + ' specificity: ' + str(specificity))
    print(' run_time: ' + str(running_time))
    print(' cnn_run_time: ' + str(cnn_time))

    print("CNN f1_mac: {}, CNN f1_mic: {}, CNN f1_wght: {}".format(f1_mac, f1_mic, f1_wght))
    print("CNN sensitivity_mac: {}, CNN sensitivity_mic: {}, CNN sensitivity_wght: {}".format(sensitivity_mac, sensitivity_mic, sensitivity_wght))


if __name__ == "__main__":
    args = run_shell_options()
    main_func(args)