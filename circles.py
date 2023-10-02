import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from matplotlib import cm
from IPython.display import Image
from tqdm import tqdm
import pandas as pd

def make_circles_simple(
    xy0=None, 
    r=1,
    n_circles=10,
    x_bound=10,
    y_bound=10,
    plus=0.5,
    img_size=128
):
    xxyy = make_mesh(-x_bound, y_bound, img_size)
    #print(f'...Mesh shape: {xxyy.shape}...')
    stacks = []
    if xy0 is None:
        low = x_bound - (plus + r / 2)
        high = y_bound - (plus + r / 2)
        xy0 = np.random.uniform(low=-low, high=high, size=(n_circles, 2))
    for x0, y0 in tqdm(xy0):
        func = lambda x: ((x[..., 0] - x0) ** 2 + (x[..., 1] - y0) ** 2) <= r**2
        stacks.append(np.apply_along_axis(func, 2, xxyy))
    return np.clip(np.stack(stacks, axis=-1).sum(axis=-1), 0, 1).astype(np.uint8)

def make_mesh(x_b, y_b, size=128):
    xx, yy = np.meshgrid(np.linspace(-x_b, x_b, size), np.linspace(-y_b, y_b, size))
    return np.concatenate([np.expand_dims(xx, axis=-1), np.expand_dims(yy, axis=-1)], axis=2)

def make_dataset(
    n_examples=20, 
    min_n_circles=1, 
    max_n_circles=10, 
    size=128
):
    y = np.random.randint(min_n_circles, max_n_circles + 1, (n_examples, 1))
    
    imgs = [make_circles_simple(n_circles=n.item(), img_size=size) for n in tqdm(y)]
    imgs = np.stack(imgs, axis=-1)
    imgs = imgs.transpose(2, 0, 1)
    imgs = np.expand_dims(imgs, axis=-1)
    return imgs, y.reshape(-1, 1)

# Complex circles functions
def make_circles_enhance(
    xy0=None, 
    r0=None, 
    min_r=0.5, 
    max_r=1.0, 
    n_circles=10, 
    x_bound=10, 
    y_bound=10, 
    extra_bound_separation=2, 
    img_size=128, 
    extra_circle_separation=0.1, 
    max_gray=0.2
):
    xxyy = make_mesh(-x_bound, y_bound, img_size)
    #print(f'...Mesh shape: {xxyy.shape}...')
    stacks = []
    new_center_tries = 10
    _on_random_r = False
    if r0 is None:
        _on_random_r = True

    if xy0 is None:
        max_radius = 1
        low = x_bound - (extra_bound_separation + max_radius / 2)
        high = y_bound - (extra_bound_separation + max_radius / 2)
        centers = np.random.uniform(low=-low, high=high, size=(n_circles, 2))
    func = lambda x: ((x[..., 0] - x0) ** 2 + (x[..., 1] - y0) ** 2) <= r**2

    for idx, (x0, y0) in enumerate(centers):
        if _on_random_r:
            r0 = np.random.rand(1).item()
            r0 = change_interval(r0, [0, 1], [min_r, max_r])
        r = 2 * r0 + extra_circle_separation

    c = 0
    while any([func(x) for i, x in enumerate(centers) if i != idx]):
        if c >= new_center_tries:
            break
        #print('\n...Center inside other circle. Assigning new center...')
        center_aux = np.random.uniform(low=-low, high=high, size=(2, ))
        x0, y0 = center_aux
        centers[idx] = center_aux
        if _on_random_r:
            r0 = np.random.rand(1).item()
            r0 = change_interval(r0, [0, 1], [min_r, max_r])
        c += 1
        r = r0
        stacks.append(np.apply_along_axis(func, 2, xxyy))
    stacks = np.stack(stacks, axis=-1).sum(axis=-1)
    stacks = np.clip(stacks, 0, 1).astype(np.uint8)
    return max_gray * stacks * np.random.rand(*stacks.shape)

def make_data_enhance(n_examples=20, 
                      min_n_circles=1, 
                      max_n_circles=10, 
                      size=128, 
                      on_random_n_circles=False):
    if on_random_n_circles:
        y = np.random.randint(min_n_circles, max_n_circles + 1, (n_examples, 1))
    else:
        inter = (max_n_circles - min_n_circles) + 1
        y = [x1 for x1 in range(min_n_circles, max_n_circles + 1) for _ in range(n_examples // inter)]
        y = np.array(y).reshape(-1, 1)

    imgs = [make_circles_enhance(n_circles=n.item(), img_size=size) for n in tqdm(y)]
    imgs = np.stack(imgs, axis=-1)
    imgs = imgs.transpose(2, 0, 1)
    imgs = np.expand_dims(imgs, axis=-1)
    return imgs, y.reshape(-1, 1)