import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import seaborn as sn
import PIL
from sklearn.metrics import confusion_matrix

from src.helpers import EMOTIONS_TO_ID, to_cpu, to_t


def plot_losses(train_losses, validation_losses):
    """Plot training and validation losses"""

    plt.plot(train_losses, '-bx')
    plt.plot(validation_losses, '-rx')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs Epoch #');


def plot_accuracies(train_accuracies, validation_accuracies):
    """Plot training and validation accuracies"""

    plt.plot(train_accuracies, '-bx')
    plt.plot(validation_accuracies, '-rx')
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'])
    plt.title('Accuracy vs Epoch #')


def display_confusion(true_and_predicted_labels):
    """Plot confusion matrix for model predictions"""

    cfm = confusion_matrix(true_and_predicted_labels["true"], true_and_predicted_labels["predictions"])
    # normalize data
    cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]

    cfm_df = pd.DataFrame(cfm, index=[emotion for emotion in EMOTIONS_TO_ID.values()],
                          columns=[emotion for emotion in EMOTIONS_TO_ID.values()])

    plt.figure(figsize=(12, 7))
    sn.heatmap(cfm_df, annot=True)


def display_incorrect(incorrect_predictions):
    """Display incorrect predictions of model"""

    incorrect_predictions["labels"] = [label.item() for label in incorrect_predictions["labels"][0]]
    incorrect_predictions["predictions"] = [prediction.item() for prediction in incorrect_predictions["predictions"][0]]

    # print the first ten images that were wrongly classified
    for i in range(10):
        print("Prediction: " + EMOTIONS_TO_ID[incorrect_predictions["predictions"][i]])
        print("Label: " + EMOTIONS_TO_ID[incorrect_predictions["labels"][i]])
        plt.imshow(to_cpu(incorrect_predictions["images"][i][0]).numpy()[0], cmap='gray')
        plt.show()


def display_personal_test(facial_expression_model, validation_test_data_transforms):
    """Classify personally uploaded images"""

    imgs = []
    for i in range(7):
        img = to_t(validation_test_data_transforms(PIL.Image.open('data/my_emotions/' + str(i) + '.png')))
        imgs.append(img)

    imgs = torch.stack(imgs)
    predictions = torch.argmax(facial_expression_model(imgs), 1)
    predictions = [prediction.item() for prediction in predictions]
    imgs = to_cpu(imgs)

    for i, prediction in enumerate(predictions):
        print(EMOTIONS_TO_ID[prediction])
        plt.imshow(imgs.numpy()[i][0], cmap='gray')
        plt.show()
