import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import PIL

from src.helpers import EMOTIONS_TO_ID, createFolders


def download_dataset():
    """Read CSV of dataset from Kaggle"""
    return pd.read_csv(os.path.abspath('icml_face_data.csv'))


def get_image_by_class(df, img_class):
    """Return an image with a label matching input label"""

    img_row = None
    for i in range(len(df) - 1, 0, -1):
        if df['emotion'][i] == img_class:
            img_row = df.iloc[i]
            break

    img = np.fromstring(img_row[' pixels'], sep=' ')
    img = np.reshape(img, (48, 48))

    return img


def display_images():
    """Print pandas df of dataset and plot one image for each unique label"""

    icml_faces = download_dataset()
    print(icml_faces.head())
    print()

    plt.figure(0, figsize=(25, 100))

    for i in range(len(EMOTIONS_TO_ID)):
        plt.subplot(1, 8, i + 1)
        plt.title(EMOTIONS_TO_ID[i])
        plt.imshow(get_image_by_class(icml_faces, i), cmap='gray')
    plt.show()


def save_data(df, set_subfolder):
    """Convert images in df to PIL form and save to project directory under appropriate folder"""

    imgs = []

    for i in df.index:
        img = np.fromstring(df.loc[i, ' pixels'], sep=' ')
        img = np.reshape(img, (48, 48))
        image = PIL.Image.fromarray(img.astype(np.uint8))
        imgs.append(image)
        image.save(os.path.abspath('data/images/' + set_subfolder + "/" + str(df['emotion'][i]) + "/" + str(i) +
                                   '.png'))


def prepare_data():
    """Download and save images in dataset"""

    createFolders()

    data_set = download_dataset()

    train_data = data_set[data_set[' Usage'] == "Training"]
    validation_data = data_set[data_set[' Usage'] == "PublicTest"]
    test_data = data_set[data_set[' Usage'] == "PrivateTest"]

    save_data(train_data, 'training')
    save_data(validation_data, 'validation')
    save_data(test_data, 'test')
