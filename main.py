import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.metrics import f1_score
from utils.utils import b_fft, Logger
from all_args import run_shell_options
from Transformer import ImageTransformer
from sklearn.metrics import accuracy_score
from keras.layers.convolutional import Conv2D
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Dense, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

plt.rcParams["font.family"] = "Times New Roman"
warnings.simplefilter(action="ignore", category=FutureWarning)


def main_func():
    args = run_shell_options()
    fileName = args.run_dataset

    # save Terminal outputs
    out_dir = args.out_dir + '/' + args.run_dataset.split('.')[0] + '/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log_file_name = out_dir+'Terminal.log'
    # record normal print messages
    sys.stdout = Logger(log_file_name)
    # record traceback error messages
    sys.stderr = Logger(log_file_name)

    """
    #################################### Stage 1 #####################################
    Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc.
    """
    print("#################### Stage 1: Data preparation #####################")
    
    # Get all datasets in list
    fileNames = []
    file_dir = args.data_dir
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file != ".DS_Store":
                fileNames.append(file)

    print("Dataset list: {}".format(fileNames))
    print("Chosen file: " + fileName)

    # Read data
    if "RowCount" in fileName:
        filepath = "Datasets/RowCount/" + fileName
        normalizedDataset = pd.read_csv(
            filepath, index_col=0, header=None, dtype="unicode"
        ).T
        normalizedDataset = normalizedDataset.rename(
            columns={normalizedDataset.columns[0]: "Unnamed: 0"}
        )
    elif "-RawCount-new" in fileName:
        filepath = "Datasets/RowCount/new/" + fileName
        normalizedDataset = pd.read_csv(
            filepath, index_col=0, header=None, dtype="unicode"
        ).T
        normalizedDataset = normalizedDataset.rename(
            columns={normalizedDataset.columns[0]: "Unnamed: 0"}
        )
    else:
        filepath = "Datasets/RawCount/" + fileName
        normalizedDataset = pd.read_csv(filepath)

    # Data clean and normalization
    normalizedDataset.drop(columns=["Unnamed: 0"], inplace=True)
    normalizedDataset.dropna(inplace=True)
    NA_rem_indx = []
    for col in normalizedDataset.columns:
        if "NA" in col:
            NA_rem_indx.append(normalizedDataset.columns.get_loc(col))
    normalizedDataset.drop(
        normalizedDataset.columns[[NA_rem_indx]], axis=1, inplace=True
    )

    normalizedDatasetT = normalizedDataset.transpose()
    labels = np.array(normalizedDataset.columns.tolist())
    for i in range(len(labels)):
        name = labels[i].split(".")
        if len(name) > 1:
            labels[i] = "".join([str(elem) for elem in name[:-1]])
        else:
            labels[i] = name[0]
    data = normalizedDatasetT.values
    for m in range(len(data)):
        data[m] = [np.int64(i) for i in data[m]]
    data = data.astype("int64")

    print("len of unique classes: {}".format(str(len(list(set(labels))))))
    print("Input Type: {}".format(str(type(data[0][0]))))

    # Train Test split
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=23, stratify=labels
    )

    print("Shape of X_train:{}".format(X_train.shape))
    print("Shape of X_test:{}".format(X_test.shape))
    print("Shape of y_train:{}".format(y_train.shape))
    print("Shape of y_test:{}".format(y_test.shape))

    # Prepare for frames
    classes = list(set(y_train))
    train_frames = []
    test_frames = []
    for class_name in classes:
        train_frame = []
        for i in range(len(y_train)):
            if class_name == y_train[i]:
                train_frame.append(X_train[i])
        train_frames.append(train_frame)

        test_frame = []
        for i in range(len(y_test)):
            if class_name == y_test[i]:
                test_frame.append(X_test[i])
        test_frames.append(test_frame)

    '''
    #################################### Stage 2 #####################################
    Stage 2: Apply Transformer and get fitters
    '''
    print("#################### Stage 2: Apply Transformer #####################")
    fitters = []
    X_train_Trans = []
    for i in range(len(classes)):
        it = ImageTransformer(
            feature_extractor="tsne", pixels=100, random_state=12345, n_jobs=-1
        )

        train_frame = np.array(train_frames[i])

        _ = it.fit(train_frame, plot=False)

        fitters.append(it)

    # Prepare train images
    true_label = []
    Pixel = 100
    x_train_img_fft = np.zeros((len(X_train), Pixel, len(classes) * Pixel, 3))

    for j in range(1):
        train_frame = X_train[j]
        class_data = []
        block_img_0 = []
        block_img_1 = []
        block_img_2 = []
        for it in fitters:
            x_train_img = it.transform(train_frame)
            block_img_0.append(x_train_img[0, :, :, 0])
            x_train_fft_real, x_train_fft_imag = b_fft(train_frame, it)
            block_img_1.append(x_train_fft_real[0, :, :, 0])
            block_img_2.append(x_train_fft_imag[0, :, :, 0])

        block_img_0 = np.concatenate(block_img_0, axis=1)
        block_img_1 = np.concatenate(block_img_1, axis=1)
        block_img_2 = np.concatenate(block_img_2, axis=1)

        x_train_img_fft[j, :, :, 0] = block_img_0
        x_train_img_fft[j, :, :, 1] = block_img_1
        x_train_img_fft[j, :, :, 2] = block_img_2

    x_train_img_fft[x_train_img_fft > 1] = 1
    X_train_imgs = x_train_img_fft

    train_label = []
    for i_label in y_train:
        for i in range(len(classes)):
            if classes[i] == i_label:
                train_label.append(i)
    y_train_imgs = np.array(train_label)

    # Prepare test images
    y_predict_label = []
    true_label = []
    test_Res = []
    x_test_img_fft = np.zeros((len(X_test), Pixel, len(classes) * Pixel, 3))

    for j in range(1):
        test_frame = X_test[j]
        class_data = []
        block_img_0 = []
        block_img_1 = []
        block_img_2 = []
        for it in fitters:
            x_test_img = it.transform(test_frame)
            block_img_0.append(x_test_img[0, :, :, 0])
            x_test_fft_real, x_test_fft_imag = b_fft(test_frame, it)
            block_img_1.append(x_test_fft_real[0, :, :, 0])
            block_img_2.append(x_test_fft_imag[0, :, :, 0])

        block_img_0 = np.concatenate(block_img_0, axis=1)
        block_img_1 = np.concatenate(block_img_1, axis=1)
        block_img_2 = np.concatenate(block_img_2, axis=1)

        x_test_img_fft[j, :, :, 0] = block_img_0
        x_test_img_fft[j, :, :, 1] = block_img_1
        x_test_img_fft[j, :, :, 2] = block_img_2

    x_test_img_fft[x_test_img_fft > 1] = 1
    X_test_imgs = x_test_img_fft

    test_label = []
    for i_label in y_test:
        for i in range(len(classes)):
            if classes[i] == i_label:
                test_label.append(i)

    print("Shape of X_train_imgs: " + str(X_train_imgs.shape))
    print("Shape of X_test_imgs: " + str(X_test_imgs.shape))

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

    # CNN model
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=in_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(len(classes)))
    model.add(Activation("sigmoid"))
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    earlyStopping = EarlyStopping(
        monitor=monitor, patience=es_patience, verbose=verbose, mode=mode
    )
    mcp_save = ModelCheckpoint(
        ".mdl_wts.hdf5", save_best_only=save_best_only, monitor=monitor, mode=mode
    )
    reduce_lr_loss = ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=lr_patience,
        verbose=verbose,
        min_delta=lr,
        mode=mode,
    )
    # model.summary()

    model.fit(
        X_train_imgs,
        y_train_imgs,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
        validation_split=validation_split,
    )

    print("Training Done.")

    '''
    #################################### Stage 4 #####################################
    Stage 4: Evaluation
    '''
    print("#################### Stage 4: Evaluation #####################")
    y_pred_0 = model.predict(X_test_imgs)
    y_predict_label = []
    for i in y_pred_0:
        y_predict_label.append(i.argmax())

    acc = accuracy_score(y_predict_label, test_label)
    f1 = f1_score(y_predict_label, test_label, average="macro")

    print(fileName + " Accuracy: " + str(acc))
    print(fileName + " F1-score: " + str(f1))


if __name__ == "__main__":
    main_func()
