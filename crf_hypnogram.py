from models import get_model_cnn, get_model_lstm, get_model_cnn_crf, get_model_bilstm_crf
import numpy as np
from utils import gen, chunker, rescale_array
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, classification_report
from glob2 import glob
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics

base_path = "data/physionet_sleep/eeg_fpz_cz"

files = sorted(glob(os.path.join(base_path, "*.npz")))

ids = sorted(list(set([x.split("/")[-1][:5] for x in files])))
print(ids)
# split by test subject
train_ids, test_ids = train_test_split(ids, test_size=0.15, random_state=1338)
#train_ids, test_ids = ['SC405', 'SC408', 'SC402', 'SC417', 'SC415', 'SC407', 'SC401', 'SC413', 'SC416', 'SC409', 'SC404', 'SC411', 'SC419', 'SC410', 'SC412', 'SC418', 'SC414', 'SC403'], ['SC406', 'SC400']

print(train_ids)
print(test_ids)
train_val, test = [x for x in files if x.split("/")[-1][:5] in train_ids], [x for x in files if
                                                                            x.split("/")[-1][:5] in test_ids]

train, val = train_test_split(train_val, test_size=0.1, random_state=1337)

train_dict = {k: np.load(k) for k in train}
test_dict = {k: np.load(k) for k in test}
val_dict = {k: np.load(k) for k in val}

model = get_model_cnn_crf()
file_path = "fpz_crf.h5"

checkpoint = ModelCheckpoint(file_path, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_crf_viterbi_accuracy", mode="max", patience=15, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_crf_viterbi_accuracy", mode="max", patience=10, verbose=2)
callbacks_list = [checkpoint, redonplat, early]
model.fit_generator(gen(train_dict, aug=False), validation_data=gen(val_dict), epochs=100, verbose=2, steps_per_epoch=1000, validation_steps=300, callbacks=callbacks_list)
model.load_weights(file_path)

preds = []
gt = []

for record in tqdm(test_dict):
    all_rows = test_dict[record]['x']
    record_y_gt = []
    record_y_pred = []
    for batch_hyp in chunker(range(all_rows.shape[0])):
        X = all_rows[min(batch_hyp):max(batch_hyp) + 1, ...]
        Y = test_dict[record]['y'][min(batch_hyp):max(batch_hyp) + 1]

        X = np.expand_dims(X, 0)

        X = rescale_array(X)

        Y_pred = model.predict(X)
        Y_pred = Y_pred.argmax(axis=-1).ravel().tolist()

        gt += Y.ravel().tolist()
        preds += Y_pred

        record_y_gt += Y.ravel().tolist()
        record_y_pred += Y_pred

    fig_1 = plt.figure(figsize=(40, 8))
    labels = ['Wake', 'N1', 'N2', 'N3', 'REM']
    ys = [0.0, 1.0, 2.0, 3.0, 4.0]
    plt.rcParams['font.size'] = '20'
    plt.plot(record_y_gt, label='Original')
    plt.plot(record_y_pred, label='Predicted')
    plt.title("Sleep Stages", fontsize=20)
    plt.yticks(ys, labels, fontsize=20)
    plt.ylabel("Classes", fontsize=20)
    plt.xlabel("Time", fontsize=20)
    plt.legend()
    plt.show()
    plt.savefig(str(test_ids) + "original_sleep_stage_LSTM.png")

f1 = f1_score(gt, preds, average="macro")

print("Seq Test f1 score : %s " % f1)

acc = accuracy_score(gt, preds)

print("Seq Test accuracy score : %s " % acc)

print(classification_report(gt, preds))

print(" 0: W, 1: N1, 2: N2, 3: N3, 4: REM")

print(metrics.confusion_matrix(gt, preds))
