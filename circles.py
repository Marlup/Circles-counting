import numpy as np
from tqdm import tqdm
import h5py
import os
from multiprocessing import Pool
import pandas as pd
from typing import Tuple, List, Optional

# ============================== #
#   Multiprocesamiento limpio   #
# ============================== #
def unpack_args(args):
    n, kwargs = args
    return safe_make_complex_dataset(n, **kwargs)

def run_multiprocess(func, n_processes, args_list):
    from multiprocessing import Pool
    with Pool(processes=n_processes) as pool:
        return pool.map(unpack_args, args_list)

def structure_data(data):
    inputs, labels = zip(*data)

    # Asegurar que todos tengan las mismas dimensiones correctas
    inputs = [np.asarray(x) for x in inputs if x is not None]
    labels = [np.asarray(l).reshape(-1, 1) for l in labels if l is not None]

    return np.concatenate(inputs, axis=0), np.concatenate(labels, axis=0)

# ============================== #
#     Dataset simple (círculos) #
# ============================== #
def make_simple_circle(
    r: float = 1.0,
    n_circles: int = 10,
    x_bound: float = 10.0,
    y_bound: float = 10.0,
    plus: float = 0.5,
    img_size: int = 128
) -> np.ndarray:
    xxyy = _make_mesh(x_bound, y_bound, img_size)
    stacks = []
    offset = plus + r / 2
    centers = np.random.uniform(low=-x_bound + offset, high=x_bound - offset, size=(n_circles, 2))
    for x_center, y_center in centers:
        is_inside_circle = lambda x: ((x[0] - x_center) ** 2 + (x[1] - y_center) ** 2) <= r**2
        stacks.append(np.apply_along_axis(is_inside_circle, 2, xxyy))
    img = np.clip(np.sum(stacks, axis=0), 0, 1).astype(np.uint8)
    return img

def _make_mesh(x_b: float, y_b: float, size: int = 128) -> np.ndarray:
    xx, yy = np.meshgrid(np.linspace(-x_b, x_b, size), np.linspace(-y_b, y_b, size))
    return np.stack((xx, yy), axis=-1)

def make_simple_dataset(
    n_examples: int = 20,
    min_n_circles: int = 1,
    max_n_circles: int = 10,
    size: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.random.randint(min_n_circles, max_n_circles + 1, size=(n_examples,))
    imgs = np.zeros((n_examples, size, size, 1), dtype=np.uint8)
    
    for i in tqdm(range(n_examples)):
        imgs[i, ..., 0] = make_simple_circle(n_circles=y[i], img_size=size)
    
    return imgs, y.reshape(-1, 1)

# ============================== #
#      Dataset complejo         #
# ============================== #
def in_circle(x: np.ndarray, c: Tuple[float, float], r: float) -> int:
    return int(((x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2) <= r ** 2)

def make_complex_circle(
    n_circles=10,
    img_size=128,
    r0=1.0,
    min_r=1.0,
    max_r=1.0,
    box_lower_bound=-10.0,
    box_higher_bound=10.0,
    max_gray=0.0,
    new_center_tries=10
) -> np.ndarray:

    xxyy = _make_mesh(box_higher_bound + max_r, box_higher_bound + max_r, img_size)
    centers = np.random.uniform(low=box_lower_bound, high=box_higher_bound, size=(n_circles, 2))

    if min_r < max_r:
        radiuses = np.random.uniform(min_r, max_r, size=(n_circles,))
    else:
        radiuses = np.full(n_circles, r0)
        radiuses = _change_interval(radiuses, [0, 1], [min_r, max_r])

    img = np.zeros((img_size, img_size), dtype=np.float32)

    for idx, ((cx, cy), r) in enumerate(zip(centers, radiuses)):
        tries = 0
        while any(in_circle(p, (cx, cy), r) for i, p in enumerate(centers) if i != idx) and tries < new_center_tries:
            cx, cy = np.random.uniform(box_lower_bound, box_higher_bound, size=(2,))
            r = np.random.uniform(min_r, max_r)
            tries += 1

        circle_mask = np.apply_along_axis(in_circle, 2, xxyy, (cx, cy), r)
        img = np.maximum(img, circle_mask)

    img = np.clip(img, 0, 1)
    if max_gray > 0.0:
        img *= max_gray * np.random.rand(*img.shape)
    img = (img * 255).astype(np.uint8)
    return img

def _change_interval(x: np.ndarray, ab: List[float], cd: List[float]) -> np.ndarray:
    a, b = ab
    c, d = cd
    return c + (x - a) * (d - c) / (b - a)

def make_complex_dataset(
    total_samples: int,
    img_size: int = 128,
    min_n_circles: int = 1,
    max_n_circles: int = 10,
    on_balanced: bool = True,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:

    n_classes = max_n_circles - min_n_circles + 1
    if on_balanced:
        samples_per_class = total_samples // n_classes
        y = np.repeat(np.arange(min_n_circles, max_n_circles + 1), samples_per_class)
        remainder = total_samples - len(y)
        if remainder > 0:
            extra_labels = np.random.choice(np.arange(min_n_circles, max_n_circles + 1), size=remainder)
            y = np.concatenate([y, extra_labels])
    else:
        y = np.random.randint(min_n_circles, max_n_circles + 1, size=(total_samples,))
    
    n_samples = len(y)
    imgs = np.zeros((n_samples, img_size, img_size, 1), dtype=np.uint8)
    
    for i in tqdm(range(n_samples)):
        imgs[i, ..., 0] = make_complex_circle(n_circles=y[i], img_size=img_size, **kwargs)
    
    return imgs, y.reshape(-1, 1)

def generate_dataset_multiprocess(dataset_func, total_samples, n_processes, *args):
    samples_per_proc = total_samples // n_processes
    args_list = [(samples_per_proc, *args) for _ in range(n_processes)]
    datasets = run_multiprocess(dataset_func, n_processes, args_list)
    return structure_data(datasets)

# ============================== #
#      Dataset con ruido        #
# ============================== #
def build_variate_dataset(
    root_path: str,
    filename: str,
    noises_strength: List[float] = [0.05],
    on_shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    
    images_gray, y = read_dataset(root_path, filename)
    stacks = [images_gray] + [images_gray + noise * np.random.rand(*images_gray.shape) for noise in noises_strength]
    
    # Concatenamos imágenes y sus versiones invertidas
    images = np.concatenate(stacks, axis=0)
    images_inverted = 1.0 - images
    images_concat = np.concatenate([images, images_inverted], axis=0)
    
    y_concat = np.tile(y, (2 * len(stacks), 1))

    if on_shuffle:
        indices = np.random.permutation(images_concat.shape[0])
        images_concat = images_concat[indices]
        y_concat = y_concat[indices]

    return np.clip(images_concat, 0.0, 1.0), y_concat

def read_dataset(root_path: str, filename: str) -> Tuple[np.ndarray, np.ndarray]:
    path = os.path.join(root_path, filename)
    with h5py.File(path, "r") as f:
        images_gray = f["images"][:]
        target_gray = f["target"][:]
    print("Shape of extracted images:", images_gray.shape)
    print("Shape of extracted targets:", target_gray.shape)
    return images_gray, target_gray

# ============================== #
#         Análisis simple       #
# ============================== #
def df_evaluation(error: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        'error': error.reshape(-1),
        'pred': np.round(pred).astype('uint8').reshape(-1)
    }).sort_values("pred")


# ============================== #
#         Safe run               #
# ============================== #
def safe_make_complex_dataset(*args, **kwargs):
    try:
        return make_complex_dataset(*args, **kwargs)
    except Exception as e:
        print(f"[Worker error - complex dataset] {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_dataset_multiprocess(dataset_func, total_samples, n_processes, **kwargs):
    base = total_samples // n_processes
    remainders = total_samples % n_processes
    samples_list = [base + (1 if i < remainders else 0) for i in range(n_processes)]

    args_list = [(n, kwargs) for n in samples_list]

    datasets = run_multiprocess(dataset_func, n_processes, args_list)
    datasets = [d for d in datasets if d[0] is not None and d[1] is not None]
    if not datasets:
        raise RuntimeError("All worker processes failed.")
    return structure_data(datasets)