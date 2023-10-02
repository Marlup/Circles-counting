import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from matplotlib import cm
from IPython.display import Image
from tqdm import tqdm
import pandas as pd

def predict_and_errors(model, X, Y, on_show=True, on_return=False):
    pred = model.predict(X)
    error = 100 * np.abs(Y - pred) / Y
    if on_show:
        fig, axes = plt.subplots(len(Y), 1, figsize=(12, 12))
        axes = axes.ravel()

    if on_show:
        for x, yt, yp, e, ax in zip(X, Y, pred, error, axes):
            ax.imshow(x[..., 0], cmap='Greys_r')
            yp = np.round(yp.item(), 2)
            e = np.round(e.item(), 2)
            ax.set_title(f'True: {yt.item()} | Prediction: {yp} | Error(Mape): {e}')
        plt.tight_layout()
    if on_return:
        return pred, error
    else:
        return None, None

def display_images(X, Y, n_examples=5, on_shuffle=False):
    Y_ = Y[:n_examples]
    X_ = X[:n_examples]
    if on_shuffle:
        np.random.shuffle(X_)
        np.random.shuffle(Y_)
    fig, axes = plt.subplots(len(X_), 1, figsize=(12, 12))
    axes = axes.ravel()

    for x, y, ax in zip(X_, Y_, axes):
        ax.imshow(x[..., 0], cmap='Greys_r')
        ax.set_title(f'Circles: {y.item()}')
    plt.tight_layout()

def plot_history(h, key_train='loss', key_val='val_loss', lim=[0, 150, 0, 25], on_mav=False, w=5):
    if on_mav:
        t = moving_average(h.history[key_train], w)
        v = moving_average(h.history[key_val], w)
    else:
        t = h.history[key_train]
        v = h.history[key_val]
    plt.plot(t, label='training')
    plt.plot(v, label='validation')
    plt.axis(lim)
    plt.legend()
    plt.grid()

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def read_image(path):
    img = utils.read_image(path,
                         grayscale=True,
                         color_mode='rgb',
                         target_size=(224, 224),
                         interpolation='nearest',
                         keep_aspect_ratio=True
                         )
    img = utils.img_to_array(img, data_format='channels_last', dtype=tf.float64)
    return img