import numpy as np
from tqdm import tqdm
import h5py
import os
from multiprocessing import Pool

def run_multiprocess(func, n_processes, *args):
    pool = Pool(processes=n_processes)

    results = []
    processes = []
    
    for _ in range(n_processes):
        process = pool.apply_async(func, args)
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        result = process.get()
        results.append(result)

    # Close and join the pool
    pool.close()
    pool.join()

    return results

def structure_data(data):
    inputs, labels = [], []
    
    for x, l in data:
        inputs.append(x)
        labels.append(l)
    return np.concatenate(inputs, 0), np.concatenate(labels, 0)

def make_simple_circle(
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
    lower_bound = x_bound - (plus + r / 2)
    higher_bound = y_bound - (plus + r / 2)
    centers = np.random.uniform(low=-lower_bound, high=higher_bound, size=(n_circles, 2))
    for x_center, y_center in centers:
        is_inside_circle = lambda x: ((x[..., 0] - x_center) ** 2 + (x[..., 1] - y_center) ** 2) <= r**2
        stacks.append(np.apply_along_axis(is_inside_circle, 2, xxyy))
    return np.clip(np.stack(stacks, axis=-1).sum(axis=-1), 0, 1).astype(np.uint8)

def make_mesh(x_b, y_b, size=128):
    xx, yy = np.meshgrid(np.linspace(-x_b, x_b, size), np.linspace(-y_b, y_b, size))
    return np.concatenate([np.expand_dims(xx, axis=-1), np.expand_dims(yy, axis=-1)], axis=2)

def make_simple_dataset(
    n_examples=20, 
    min_n_circles=1, 
    max_n_circles=10, 
    size=128
):
    y = np.random.randint(min_n_circles, max_n_circles + 1, (n_examples, 1))
    
    imgs = [make_simple_circle(n_circles=n.item(), img_size=size) for n in tqdm(y)]
    imgs = np.stack(imgs, axis=-1)
    imgs = imgs.transpose(2, 0, 1)
    imgs = np.expand_dims(imgs, axis=-1)
    return imgs, y.reshape(-1, 1)

def in_circle(x, c, r):
    return int(((x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2) <= r**2)

# Functions for images of complex circles 
def make_complex_circle(
    n_circles=10,
    img_size=128,
    r0=None, 
    min_r=0.5, 
    max_r=2.0, 
    box_lower_bound=-10.0, 
    box_higher_bound=10.0, 
    max_gray=0.2,
    new_center_tries=10
    ):
    
    box_extra_lower_bound = box_lower_bound - max_r
    box_extra_higher_bound = box_higher_bound + max_r

    xxyy = make_mesh(-box_extra_lower_bound, 
                     box_extra_higher_bound, 
                     img_size)
    #print(f'...Mesh shape: {xxyy.shape}...')
    _on_random_r = False
    if r0 is None:
        _on_random_r = True

    if _on_random_r:
        centers = np.random.uniform(low=box_lower_bound, 
                                    high=box_higher_bound, 
                                    size=(n_circles, 2))

    if _on_random_r:
        radiuses = np.random.uniform(min_r, max_r, size=(n_circles, ))
    else:
        radiuses = np.repeat(r0, repeats=n_circles).reshape(-1, 1)
        radiuses = change_interval(radiuses, [0, 1], [min_r, max_r])

    img_update = np.zeros((img_size, img_size))
    for idx, ((c_x, c_y), radius) in enumerate(zip(centers, radiuses)):
        img = np.apply_along_axis(in_circle, 2, xxyy, (c_x, c_y), radius)
        
        c = 0
        while any([in_circle(p, (c_x, c_y), radius) for i, p in enumerate(centers) if i != idx]):
            
            if c >= new_center_tries:
                break
            c_x, c_y = np.random.uniform(low=box_lower_bound, 
                                           high=box_higher_bound, 
                                           size=(2, ))
            if _on_random_r:
                radius = np.random.uniform(min_r, max_r, 1).item()
            img = np.apply_along_axis(in_circle, 2, xxyy, (c_x, c_y), radius)
            c += 1
        img_update += img
    img_update = np.clip(img_update, 0, 1).astype(np.uint8)
    if max_gray > 0.0:
        return max_gray * img_update * np.random.rand(*img_update.shape)
    return img_update

def change_interval(x, ab=None, cd=[0, 1]):
    if ab is not None:
        a, b = ab
        c, d = cd
        xp = c + (x - a) * (d - c) / (b - a)
        return xp
    else:
        return x

def make_complex_dataset(n_min_examples=20, 
                         size=128,
                         min_n_circles=1, 
                         max_n_circles=10, 
                         on_balanced=True,
                         *others
                         ):
    if on_balanced:
        y = np.repeat(np.arange(min_n_circles, max_n_circles + 1), n_min_examples)
    else:
        y = np.random.randint(min_n_circles, max_n_circles + 1, (n_min_examples, ))
    # others: r0, min_r, max_r, box_lower_bound, box_higher_bound, max_gray, new_center_tries 
    imgs = np.array([make_complex_circle(n_circles=n, img_size=size, *others) for n in tqdm(y)])
    imgs = np.expand_dims(imgs, axis=-1)
    return imgs, y

def build_variate_dataset(root_path,
                          file_name,
                          noises_strength=[0.05],
                          on_shuffle=True,
                          ):
    images_gray, y = read_dataset(root_path, file_name)
    stacks = []
    stacks.append(images_gray)
    for noise_strength in noises_strength:
        stacks.append(images_gray + noise_strength * np.random.rand(*images_gray.shape))
    images_concat = np.concatenate(stacks, axis=0)
    images_values_inverted = np.ones(images_concat.shape) - images_concat
    images_concat = np.concatenate([images_concat,
                                  images_values_inverted], axis=0)
    n_stacks = 2 * len(stacks)
    y_concat = np.concatenate([y] * n_stacks,
                              axis=0)
    if on_shuffle:
        indices = np.arange(len(images_concat))
        np.random.shuffle(indices)
        images_concat = images_concat[indices]
        y_concat = y_concat[indices]
    return np.clip(images_concat, 0.0, 1.0), y_concat

def read_dataset(root_path, file_name):
    file_path = os.path.join(root_path, file_name)
    with h5py.File(file_path, "r") as f:
        images_gray = f.get('images')[:]
        target_gray = f.get('target')[:]
    print("Shape of extracted images:", images_gray.shape)
    print("Shape of extracted targets:", target_gray.shape)
    return images_gray, target_gray