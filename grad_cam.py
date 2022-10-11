from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import gen, chunker, WINDOW_SIZE, rescale_array
from glob import glob
import os
from tqdm import tqdm

model = load_model('load the model here')
print(model.summary())

base_path = "enter the data here"
files = sorted(glob(os.path.join(base_path, "*.npz")))
ids = sorted(list(set([x.split("/")[-1][:5] for x in files])))

for id in ids:
    test = [x for x in files if x.split("/")[-1][:5] in ids]
    test_dict = {k: np.load(k) for k in test}


    def grad_cam(layer_name, data):
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(data)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)

        pooled_grads = tf.reduce_mean(grads, axis=1)

        last_conv_layer_output = last_conv_layer_output[0]

        heatmap = last_conv_layer_output + pooled_grads
        heatmap = tf.reduce_mean(heatmap, axis=1)
        heatmap = np.expand_dims(heatmap, 0)
        return heatmap

    preds = []
    gt = []

    for record in tqdm(test_dict):
        all_rows = test_dict[record]['x']
        print(all_rows.shape)
        record_y_gt = []
        record_y_pred = []
        for batch_hyp in chunker(range(all_rows.shape[0])):
            X = all_rows[min(batch_hyp):max(batch_hyp) + 1, ...]
            Y = test_dict[record]['y'][min(batch_hyp):max(batch_hyp) + 1]

            X = rescale_array(X)

            Y_pred = model.predict(X)
            Y_pred = Y_pred.argmax(axis=-1).ravel().tolist()

            gt += Y.ravel().tolist()
            preds += Y_pred

            record_y_gt += Y.ravel().tolist()
            record_y_pred += Y_pred

            layer_name = "enter the layer name"
            label = ['Wake', 'N1', 'N2', 'N3', 'REM']
            cnt = 0
            for i in X:
                data = np.expand_dims(i, 0)
                prediction = model.predict(data)
                pred_index = np.argmax(prediction)
                if (prediction == np.amax(prediction)).any():
                    # print('working')
                    heatmap = grad_cam(layer_name, data)
                    print(cnt)
                    print(f"Model prediction = ({pred_index}), True label = {label[int(Y[cnt])]}")
                    plt.figure(figsize=(30, 5))
                    plt.imshow(np.expand_dims(heatmap, axis=2),
                               aspect="auto",
                               cmap="Oranges",
                               alpha=0.8,
                               interpolation='bilinear',
                               extent=["enter the length of the signal (3000,1)"])
                    plt.plot(i, 'k')
                    plt.colorbar()
                    plt.show()
                    plt.savefig(str(id) + str(cnt) + "model.png")
                cnt += 1

