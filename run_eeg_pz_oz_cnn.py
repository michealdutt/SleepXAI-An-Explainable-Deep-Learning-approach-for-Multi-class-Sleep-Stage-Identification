from models import get_model_cnn, get_model_lstm, get_model_cnn_crf
import numpy as np
from utils import gen, chunker, WINDOW_SIZE, rescale_array, rescale_wake
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
from sklearn.metrics import f1_score, accuracy_score, classification_report
from glob2 import glob
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import matplotlib
matplotlib.use('agg')
from sklearn.metrics import confusion_matrix
#from keras_contrib.layers import CRF
#from keras_contrib.losses import  crf_loss
#from keras_contrib.metrics import crf_viterbi_accuracy


np.set_printoptions(precision=2)


def train_model(train_files, model_save_path):
    train, val = train_test_split(train_files, test_size=0.1)#, random_state=1337)

    train_dict = {k: np.load(k) for k in train}

    val_dict = {k: np.load(k) for k in val}
    print("Validating: " + str(val_dict))

    model = get_model_cnn()

    checkpoint = ModelCheckpoint(model_save_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=20, verbose=1)

    redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=5, verbose=2)

    callbacks_list = [checkpoint, redonplat, early]

    model.fit_generator(gen(train_dict, aug=False), validation_data=gen(val_dict), epochs=100, verbose=2,
                        steps_per_epoch=1000, validation_steps=300, callbacks=callbacks_list)

    model.save(model_save_path)


def eval_model(model, test_files):
    test_dict = {k: np.load(k) for k in test_files}
    print("Testing: " + str(test_dict))
    preds = []
    gt = []

    for record in tqdm(test_dict):
        all_rows = test_dict[record]['x']
        record_y_gt = []
        record_y_pred = []

        X = all_rows
        Y = test_dict[record]['y']
        wakeStd = rescale_wake(X, Y)
        X = np.expand_dims(X, 0)
        X = (X - np.mean(X)) / wakeStd
        # X = rescale(X, Y) #default

        Y_pred = model.predict(X)
        Y_pred = Y_pred.argmax(axis=-1).ravel().tolist()


        gtNow = Y.ravel().tolist()
        gt += gtNow
        preds += Y_pred

        record_y_gt += Y.ravel().tolist()
        record_y_pred += Y_pred

        acc_Test = accuracy_score(gtNow, Y_pred)
        f1_Test = f1_score(gtNow, Y_pred, average="macro")
        print("acc %s, f1 %s"%(acc_Test, f1_Test))
        print("Classification Report")
        print(classification_report(gtNow, Y_pred))
        print("Confusion Matrix")
        print(confusion_matrix(gtNow, Y_pred))

    return gt, preds

def cross_validation_training():

    base_path_2 = "path to eeg_pz_cz files"

    model_save_path_2 = "pz_oz_cnn.h5"

    print("Loading Data from path: %s" % (base_path_2))

    files_2 = sorted(glob(os.path.join(base_path_2, "*.npz")))

    subject_ids_2 = list(set([x.split("/")[-1][:5] for x in files_2]))

    allpreds_2 = []
    allgt_2 = []

    for i in subject_ids_2:
        test_id = {i}
        all_subjects_1 = set(subject_ids_2)

        train_ids_1 = all_subjects_1 - test_id

        train_files_1, test_files_1 = [x for x in files_2 if x.split("/")[-1][:5] in train_ids_1], \
                                      [x for x in files_2 if x.split("/")[-1][:5] in test_id]
        train_model(train_files_1, model_save_path_2)
        gt, preds = eval_model(load_model(model_save_path_2), test_files_1)
        allpreds_2 += preds
        allgt_2 += gt

    accuracy_oz = []
    print("Testing on eeg_pz_oz")
    f1_2 = f1_score(allgt_2, allpreds_2, average="macro")
    acc_2 = accuracy_score(allgt_2, allpreds_2)
    print("acc_20fold_model_2 %s, f1_20fold_model_2 %s" % (acc_2, f1_2))
    print(classification_report(allgt_2, allpreds_2))
    eeg_pz_oz_report = pd.DataFrame(classification_report(y_true = allgt_2, y_pred = allpreds_2, output_dict=True)).transpose()
    eeg_pz_oz_report.to_csv('eeg_pz_oz_report_cnn.csv', index= True)
    print("Result:- Signal eeg_pz_oz")
    print(confusion_matrix(allgt_2, allpreds_2))
    eeg_pz_oz_confusion_report = pd.DataFrame(confusion_matrix(y_true = allgt_2, y_pred = allpreds_2))
    eeg_pz_oz_confusion_report.to_csv('eeg_pz_oz_confusion_report_cnn.csv', index= True)
    accuracy_oz.append(acc_2)
    accuracy_oz = np.array(accuracy_oz)
    np.savetxt("accuracy_oz_cnn.csv", accuracy_oz, delimiter=",")

if __name__ == "__main__":
    model_path = cross_validation_training()