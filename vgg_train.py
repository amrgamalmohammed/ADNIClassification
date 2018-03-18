import os
import gc
import sys
import time
import datetime
import traceback
import pandas as pd
import numpy as np
import nibabel as nib
from tqdm import *

from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv3D
from keras.layers import MaxPooling3D
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Add
from keras.utils import plot_model

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.cross_validation import StratifiedKFold

import matplotlib.pyplot as plt

import seaborn as sns

from utils import iterate_minibatches, iterate_minibatches_train

PATH_TO_REP = 'data/'  # adni_data

input_shape = (1, 192, 192, 160)

seed = 7
np.random.seed(seed)


def train(X_train, y_train,
          X_test, y_test, architecture,
          LABEL_1, LABEL_2,  # labels of the y.
          num_epochs=100, batchsize=5,
          dict_of_paths={'output': '1.txt', 'picture': '1.png',
                         'report': 'report.txt'},
          report='''trained next architecture, used some
                    optimizstion method with learning rate...'''):
    """
    Iterate minibatches on train subset and validate results on test subset.

    Parameters
    ----------
    X_train : numpy array
        X train subset.
    y_train : numpy array
        Y train subset.
    X_test : numpy array
        X test subset.
    y_test : numpy array
        Y test subset.
    LABEL_1 : {'AD', 'LMCI', 'EMCI', 'Normal'}
        String label for target == 0.
    LABEL_2 : {'AD', 'LMCI', 'EMCI', 'Normal'}
        String label for target == 1.
    dict_of_paths : dictionary
        Names of files to store results.
    report : string
        Some comments which will saved into report after ending of training.
    num_epochs : integer
        Number of epochs for all of the experiments. Default is 100.
    batchsize : integer
        Batchsize for network training. Default is 5.

    Returns
    -------
    tr_losses : numpy.array
        Array with loss values on train.
    val_losses : numpy.array
        Array with loss values on test.
    val_accs : numpy.array
        Array with accuracy values on test.
    rocs : numpy.array
        Array with roc auc values on test.

    """

    eps = []
    tr_losses = []
    val_losses = []
    val_accs = []
    rocs = []

    FILE_PATH = dict_of_paths['output']
    PICTURE_PATH = dict_of_paths['picture']
    REPORT_PATH = dict_of_paths['report']

    # here we written outputs on each step (val and train losses, accuracy, auc)
    with open(FILE_PATH, 'w') as f:
        f.write('\n----------\n\n' + str(datetime.datetime.now())[:19])
        f.write('\n' + LABEL_1 + '-' + LABEL_2 + '\n')
        f.close()

    # starting training
    print("Starting training...")
    sys.stdout.flush()
    den = X_train.shape[0] / batchsize
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches_train(X_train, y_train, batchsize,
                                               shuffle=True):
            inputs, targets = batch
            history = architecture.fit(inputs, targets)
            train_err = train_err + np.mean(history.history['loss'])
            train_batches = train_batches + 1

        val_err = 0
        val_batches = 0
        preds = []
        targ = []
        for batch in iterate_minibatches(X_test, y_test, batchsize,
                                         shuffle=False):
            inputs, targets = batch
            err = architecture.evaluate(inputs, targets)
            val_err = val_err + np.mean(err)
            val_batches = val_batches + 1
            out = architecture.predict(inputs)
            [preds.append(i) for i in out]
            [targ.append(i) for i in targets]

        preds_tst = np.array(preds).argmax(axis=1)
        ##
        ## output
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs,
                                                   time.time() - start_time))
        sys.stdout.flush()
        print("  training loss:\t\t{:.7f}".format(train_err / train_batches))
        sys.stdout.flush()
        print("  validation loss:\t\t{:.7f}".format(val_err / val_batches))
        sys.stdout.flush()
        print('  validation accuracy:\t\t{:.7f}'.format(
            accuracy_score(np.array(targ),
                           preds_tst)))
        sys.stdout.flush()
        print('Confusion matrix for test:')
        sys.stdout.flush()
        print(confusion_matrix(np.array(targ), np.array(preds).argmax(axis=1)))
        sys.stdout.flush()
        rcs = roc_auc_score(np.array(targ), np.array(preds))
        sys.stderr.write('Pairwise ROC_AUCs: ' + str(rcs))
        print('')

        with open(FILE_PATH, 'a') as f:
            f.write("\nEpoch {} of {} took {:.3f}s".format(epoch + 1,
                                                           num_epochs,
                                                           time.time() - start_time))
            f.write(
                "\n training loss:\t\t{:.7f}".format(train_err / train_batches))
            f.write(
                "\n validation loss:\t\t{:.7f}".format(val_err / val_batches))
            f.write('\n validation accuracy:\t\t{:.7f}'.format(
                accuracy_score(np.array(targ),
                               np.array(preds).argmax(axis=1))))

            f.write('\n Pairwise ROC_AUCs:' + str(rcs) + '\n')
            f.close()
        ## output
        ## saving results
        eps.append(epoch + 1)
        tr_losses.append(train_err / train_batches)
        val_losses.append(val_err / val_batches)
        val_accs.append(
            accuracy_score(np.array(targ), np.array(preds).argmax(axis=1)))
        rocs.append(rcs)

    print('ended!')

    ### and save plots
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.title('Loss ' + LABEL_1 + ' vs ' + LABEL_2)
    plt.xlabel('Epoch')
    plt.ylim((0, 3))
    plt.ylabel('Loss')
    plt.plot(eps, tr_losses, label='train')
    plt.plot(eps, val_losses, label='validation')
    plt.legend(loc=0)
    #
    plt.subplot(2, 2, 2)
    plt.title('Accuracy ' + LABEL_1 + ' vs ' + LABEL_2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(eps, val_accs, label='validation accuracy')
    plt.legend(loc=0)
    #
    plt.subplot(2, 2, 3)
    plt.title('AUC ' + LABEL_1 + ' vs ' + LABEL_2)
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.plot(eps, np.array(rocs), label='validation auc')
    plt.legend(loc=0)
    #
    plt.subplot(2, 2, 4)
    plt.title('architecture')
    plt.axis('off')
    plt.text(0, -0.1, architecture, fontsize=7, )
    plt.savefig(PICTURE_PATH)
    ###########

    # write that trainig was ended
    with open(FILE_PATH, 'a') as f:
        f.write('\nended at ' + str(datetime.datetime.now())[:19] + '\n \n')
        f.close()

    # write report
    with open(REPORT_PATH, 'a') as f:
        f.write(
            '\n' + LABEL_1 + ' vs ' + LABEL_2 + '\n' + report)
        #         f.write(architecture)
        f.write('final results are:')
        f.write('\n tr_loss: ' + str(tr_losses[-1]) + '\n val_loss: ' + \
                str(val_losses[-1]) + '\n val_acc; ' + str(val_accs[-1]) + \
                '\n val_roc_auc: ' + str(rocs[-1]))
        f.write('\nresults has been saved in files:\n')
        f.write(FILE_PATH + '\n')
        f.write(PICTURE_PATH + '\n')
        f.write('\n ___________________ \n\n\n')
        f.close()

    return tr_losses, val_losses, val_accs, rocs


def build_ResNet():

    input = Input(input_shape)

    conv1_a = Conv3D(filters=32, kernel_size=(3, 3, 3), padding="same")(input)
    bnorm1_a = BatchNormalization()(conv1_a)
    relu1_a = Activation(activation="relu")(bnorm1_a)

    conv1_b = Conv3D(filters=32, kernel_size=(3, 3, 3), padding="same")(relu1_a)
    bnorm1_b = BatchNormalization()(conv1_b)
    relu1_b = Activation(activation="relu")(bnorm1_b)

    conv1_c = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding="same")(relu1_b)

    voxres2_bn1 = BatchNormalization()(conv1_c)
    voxres2_relu1 = Activation(activation="relu")(voxres2_bn1)

    voxres2_conv1 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding="same")(voxres2_relu1)
    voxres2_bn2 = BatchNormalization()(voxres2_conv1)
    voxres2_relu2 = Activation(activation="relu")(voxres2_bn2)

    voxres2_conv2 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding="same")(voxres2_relu2)

    voxres2_out = Add()([conv1_c, voxres2_conv2])

    voxres3_bn1 = BatchNormalization()(voxres2_out)
    voxres3_relu1 = Activation(activation="relu")(voxres3_bn1)

    voxres3_conv1 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding="same")(voxres3_relu1)
    voxres3_bn2 = BatchNormalization()(voxres3_conv1)
    voxres3_relu2 = Activation(activation="relu")(voxres3_bn2)

    voxres3_conv2 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding="same")(voxres3_relu2)

    voxres3_out = Add()([voxres3_conv2, voxres2_out])

    bn4 = BatchNormalization()(voxres3_out)
    relu4 = Activation(activation="relu")(bn4)

    conv4 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding="same")(relu4)

    voxres5_bn1 = BatchNormalization()(conv4)
    voxres5_relu1 = Activation(activation="relu")(voxres5_bn1)

    voxres5_conv1 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding="same")(voxres5_relu1)
    voxres5_bn2 = BatchNormalization()(voxres5_conv1)
    voxres5_relu2 = Activation(activation="relu")(voxres5_bn2)

    voxres5_conv2 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding="same")(voxres5_relu2)

    voxres5_out = Add()([voxres5_conv2, conv4])

    voxres6_bn1 = BatchNormalization()(voxres5_out)
    voxres6_relu1 = Activation(activation="relu")(voxres6_bn1)

    voxres6_conv1 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding="same")(voxres6_relu1)
    voxres6_bn2 = BatchNormalization()(voxres6_conv1)
    voxres6_relu2 = Activation(activation="relu")(voxres6_bn2)

    voxres6_conv2 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding="same")(voxres6_relu2)

    voxres6_out = Add()([voxres5_out, voxres6_conv2])

    bn7 = BatchNormalization()(voxres6_out)
    relu7 = Activation(activation="relu")(bn7)
    conv7 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding="same")(relu7)

    voxres8_bn1 = BatchNormalization()(conv7)
    voxres8_relu1 = Activation(activation="relu")(voxres8_bn1)

    voxres8_conv1 = Conv3D(filters=128, kernel_size=(3, 3, 3), padding="same")(voxres8_relu1)
    voxres8_bn2 = BatchNormalization()(voxres8_conv1)
    voxres8_relu2 = Activation(activation="relu")(voxres8_bn2)

    voxres8_conv2 = Conv3D(filters=128, kernel_size=(3, 3, 3), padding="same")(voxres8_relu2)

    voxres8_out = Add()([voxres8_conv2, conv7])

    voxres9_bn1 = BatchNormalization()(voxres8_out)
    voxres9_relu1 = Activation(activation="relu")(voxres9_bn1)

    voxres9_conv1 = Conv3D(filters=128, kernel_size=(3, 3, 3), padding="same")(voxres9_relu1)
    voxres9_bn2 = BatchNormalization()(voxres9_conv1)
    voxres9_relu2 = Activation(activation="relu")(voxres9_bn2)

    voxres9_conv2 = Conv3D(filters=128, kernel_size=(3, 3, 3), padding="same")(voxres9_relu2)

    voxres9_out = Add()([voxres9_conv2, voxres8_out])

    pool10 = MaxPooling3D(pool_size=(7, 7, 7), data_format="channels_first")(voxres9_out)
    den10 = Dense(units=128)(pool10)
    relu10 = Activation(activation="relu")(den10)
    flat10 = Flatten()(relu10)
    out = Dense(units=1, activation="sigmoid")(flat10)

    model = Model(inputs=input, outputs=out)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model





def build_net():
    """
    Method for VGG like net Building.
    """
    model = Sequential()
    model.add(Conv3D(8, (3, 3, 3), activation="relu", input_shape=input_shape, data_format='channels_first'))
    model.add(Conv3D(8, (3, 3, 3), activation="relu", data_format='channels_first'))
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(Conv3D(16, (3, 3, 3), activation="relu", data_format='channels_first'))
    model.add(Conv3D(16, (3, 3, 3), activation="relu", data_format='channels_first'))
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(Conv3D(32, (3, 3, 3), activation="relu", data_format='channels_first'))
    model.add(Conv3D(32, (3, 3, 3), activation="relu", data_format='channels_first'))
    model.add(Conv3D(32, (3, 3, 3), activation="relu", data_format='channels_first'))
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(Conv3D(64, (3, 3, 3), activation="relu", data_format='channels_first'))
    model.add(Conv3D(64, (3, 3, 3), activation="relu", data_format='channels_first'))
    model.add(Conv3D(64, (3, 3, 3), activation="relu", data_format='channels_first'))
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.7))
    model.add(Dense(64, activation="relu"))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

rnd_states = [14, 11, 1993, 19931411, 14111993]

def run_cross_validation(LABEL_1, LABEL_2, results_folder):
    """
    Method for cross-validation.
    Takes two labels, reading data, prepair data with this labels for trainig.

    Parameters
    ----------
    LABEL_1 : {'AD', 'LMCI', 'EMCI', 'Normal'}
        String label for target == 0.
    LABEL_2 : {'AD', 'LMCI', 'EMCI', 'Normal'}
        String label for target == 1.
    results_folder : string
        Folder to store results.

    Returns
    -------
    None.
    """
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # reading data
    gc.collect()
    metadata = pd.read_csv(PATH_TO_REP + 'metadata.csv')
    smc_mask = (
    (metadata.Label == LABEL_1) | (metadata.Label == LABEL_2)).values.astype(
        'bool')
    y = np.array((metadata[smc_mask].Label == LABEL_1).astype(np.int32).values)

    data = np.zeros((smc_mask.sum(), 1, 192, 192, 160), dtype='float32')
    # into memory
    for it, im in tqdm(enumerate(metadata[smc_mask].Path.values),
                       total=smc_mask.sum(), desc='Reading MRI to memory'):
        mx = nib.load(im).get_data().max(axis=0).max(axis=0).max(axis=0)
        data[it, 0, :, :, :] = np.array(nib.load(im).get_data()) / mx

    # loop by random states (different splitting)
    for i in range(len(rnd_states)):
        
        skf = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=rnd_states[i])

        result = open(results_folder+'run_'+str(i)+'.txt', 'w')

        for tr, ts in skf:
            X_train = data[tr]
            X_test = data[ts]
            y_train = y[tr]
            y_test = y[ts]

            #model = build_net()
            model = build_ResNet()

            train = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=5, epochs=100, verbose=1)

            sys.stdout.flush()
            result.write("\n History train loss: \n" + np.array2string(np.array(train.history['loss'])))
            result.write("\n History train acc: \n" + np.array2string(np.array(train.history['acc'])))
            result.write("\n History test loss: \n" + np.array2string(np.array(train.history['val_loss'])))
            result.write("\n History test loss: \n" + np.array2string(np.array(train.history['val_acc'])))
            result.write("\nTrain Loss: " + str(np.mean(train.history['loss'])))
            result.write("\nTrain Acc: " + str(np.mean(train.history['acc'])))
            result.write("\nTest Loss: min: " + str(np.min(train.history['val_loss'])) + " mean: " + str(np.mean(train.history['val_loss'])))
            result.write("\nTest Acc: max: " + str(np.max(train.history['val_acc'])) + " mean: " + str(np.mean(train.history['val_acc'])))
            result.write("\n######################################################")
            result.flush()

            print("## Run "+str(i)+" Finished")
            sys.stdout.flush()

        result.close()





#run_cross_validation('AD', 'Normal', './results_cnn/ad_vs_norm/')
run_cross_validation('AD', 'EMCI', './results_cnn/ad_vs_mci/')
#run_cross_validation('EMCI', 'Normal', './results_cnn/mci_vs_norm/')