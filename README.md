# Circle Counting CNN

This repository contains a Jupyter notebook for generating synthetic images with random circles and training a standard Convolutional Neural Network (CNN) model to count the circles in each image.

## Notebook Contents

- [Notebook](Circle_detection_models.ipynb). The Jupyter notebook where the image generation and CNN training are implemented.
- [Image Generation](#image-generation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Image Generation

The image generation code is provided in the `generate_images.py` file. It includes two functions for generating synthetic images with random circles:

1. `make_circles_simple`: Generates images with a random number of circles with uniform distribution.
2. `make_circles_enhance`: Generates images with a random number of circles with enhanced control over circle characteristics.

You can use these functions to create custom datasets for training and testing your circle counting CNN.

## Usage

+ Clone the repository:

 ```bash
 git clone https://github.com/your-username/circle-counting-cnn.git
```
  
+ Open and run the Jupyter notebook notebook.ipynb. This notebook demonstrates how to generate images and train a CNN model to count circles.
+ Customize the image generation parameters and model architecture as needed for your specific application.

## Dependencies
The following Python libraries are used in this project:

```bash
numpy
matplotlib
scikit-learn
tqdm
pandas
tensorFlow
```

You can install these dependencies using pip:
```bash
pip install numpy matplotlib scikit-learn tqdm pandas tensorflow
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

Feel free to explore the notebook and the image generation code to create synthetic datasets for training and evaluating your circle counting CNN.
