import argparse
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from matplotlib import cm
from IPython.display import Image
from tqdm import tqdm
import pandas as pd


def save_dataset_to_hdf5(images: np.ndarray,
                         labels: np.ndarray,
                         output_dir: str,
                         min_n_circles: int,
                         max_n_circles: int,
                         img_size: int,
                         prefix: str = "complex_images") -> str:
    """
    Guarda imágenes y etiquetas en un archivo .hdf5.

    Args:
        images (np.ndarray): Array de imágenes (N, H, W, C).
        labels (np.ndarray): Array de etiquetas (N, 1).
        output_dir (str): Carpeta donde guardar el archivo.
        min_n_circles (int): Valor mínimo de círculos (para nombre).
        max_n_circles (int): Valor máximo de círculos (para nombre).
        img_size (int): Tamaño de imagen (para nombre).
        prefix (str): Prefijo del archivo (opcional).

    Returns:
        str: Ruta completa al archivo guardado.
    """
    os.makedirs(output_dir, exist_ok=True)

    file_name = f"{prefix}_{min_n_circles}_{max_n_circles}_{img_size}x{img_size}.hdf5"
    output_file = os.path.join(output_dir, file_name)

    with h5py.File(output_file, "w") as f:
        f.create_dataset("images", data=images)
        f.create_dataset("target", data=labels)

    print(f"[✔] Dataset saved: {output_file}")
    return output_file

def load_dataset_from_file(file_path: str,
                           images_key: str = "images",
                           target_key: str = "target") -> tuple[np.ndarray, np.ndarray]:
    """
    Carga un dataset desde un archivo HDF5 dado el path y los nombres de clave.

    Args:
        file_path (str): Ruta al archivo .hdf5.
        images_key (str): Clave para acceder a los datos de imagen.
        target_key (str): Clave para acceder a los datos de etiquetas.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (images, targets)
    """
    with h5py.File(file_path, "r") as f:
        if images_key not in f or target_key not in f:
            raise KeyError(f"Missing dataset keys: {images_key} or {target_key}")
        
        images = f[images_key][:]
        targets = f[target_key][:]
    
    print(f"[✔] Loaded from: {file_path}")
    print(f"  {images_key}.shape = {images.shape}")
    print(f"  {target_key}.shape = {targets.shape}")
    return images, targets

def parse_args():
    parser = argparse.ArgumentParser(description="Generate complex circle dataset")

    parser.add_argument("--total_samples", type=int, default=100)
    parser.add_argument("--n_processes", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--min_n_circles", type=int, default=1)
    parser.add_argument("--max_n_circles", type=int, default=5)
    parser.add_argument("--on_balanced", type=bool, default=True)
    parser.add_argument("--r0", type=float, default=0.0)
    parser.add_argument("--min_r", type=float, default=1.0)
    parser.add_argument("--max_r", type=float, default=2.0)
    parser.add_argument("--box_lower_bound", type=float, default=-10.0)
    parser.add_argument("--box_higher_bound", type=float, default=10.0)
    parser.add_argument("--max_gray", type=float, default=0.2)
    parser.add_argument("--new_center_tries", type=int, default=10)
    parser.add_argument("--output_path", type=str, default="./")

    return parser.parse_args()

def predict_and_errors(model, X, Y, on_show=True, on_return=False, n_columns=4):
    import math
    pred = model.predict(X)
    error = 100 * np.abs(Y - pred) / Y

    n_samples = len(Y)
    n_rows = math.ceil(n_samples / n_columns)

    if on_show:
        fig, axes = plt.subplots(n_rows, n_columns, figsize=(4 * n_columns, 4 * n_rows))
        axes = np.array(axes).reshape(-1)  # Asegura que sea una lista 1D para iterar fácil

        for idx, (x, yt, yp, e) in enumerate(zip(X, Y, pred, error)):
            ax = axes[idx]
            ax.imshow(x[..., 0], cmap='Greys_r')
            yp_val = np.round(yp.item(), 2)
            e_val = np.round(e.item(), 2)
            ax.set_title(f'True: {yt.item()} | Pred: {yp_val} | MAPE: {e_val}%')
            ax.axis('off')

        # Desactiva ejes vacíos
        for idx in range(n_samples, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()

    if on_return:
        return pred, error
    else:
        return None, None

def display_images(X, Y, n_examples=5, n_columns=3, on_shuffle=False):
    import math
    
    # Tomar solo los primeros n_examples
    X_ = X[:n_examples]
    Y_ = Y[:n_examples]

    if on_shuffle:
        indices = np.random.permutation(len(X_))
        X_ = X_[indices]
        Y_ = Y_[indices]

    # Calcular número de filas
    n_rows = math.ceil(n_examples / n_columns)

    # Crear subplots
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(4 * n_columns, 4 * n_rows))
    axes = np.array(axes).reshape(-1)  # Aplanar matriz de ejes

    for i in range(n_rows * n_columns):
        ax = axes[i]
        if i < n_examples:
            x, y = X_[i], Y_[i]
            ax.imshow(x[..., 0], cmap='Greys_r')
            ax.set_title(f'Circles: {y.item()}')
        else:
            ax.axis('off')  # Ocultar ejes no utilizados

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