import torch

from src.helpers import to_t
from src.data.load_dataset import get_data_loaders


def calculate_accuracy(model):
    """Calculate accuracy of model on test data"""

    incorrect_predictions = {"images": [], "labels": [], "predictions": []}
    # keep track of image labels and predicted labels for confusion matrix
    true_and_predicted_labels = {"true": [], "predictions": []}

    data_loaders, _ = get_data_loaders()

    num_correct_predictions = 0
    num_predictions = 0

    for images, labels in data_loaders['Test']:
        # convert tensors to gpu if available
        images = to_t(images)
        labels = to_t(labels)

        # complete forward pass with images in current batch
        outputs = model(images)

        # calculate predicted emotions for current batch
        predictions = torch.argmax(outputs.data, 1)

        # save image labels and predictions for confusion matrix
        for i in range(len(outputs)):
            true_and_predicted_labels["true"].append(labels.cpu().numpy()[i])
            true_and_predicted_labels["predictions"].append(predictions.cpu().numpy()[i])

        # update total num of predictions and num correct predictions for current batch
        num_predictions += labels.size(0)
        num_correct_predictions += (predictions == labels).sum()

        # save images and their labels and predictions for later analysis
        incorrect_predictions["images"].append(images[predictions != labels])
        incorrect_predictions["labels"].append(labels[predictions != labels])
        incorrect_predictions["predictions"].append(predictions[predictions != labels])

    # calculate model accuracy for current epoch
    test_accuracy = 100 * num_correct_predictions / num_predictions

    print("\nFINAL RESULTS")
    print(f"Model accuracy on test data: {test_accuracy:.4f}%")

    return incorrect_predictions, true_and_predicted_labels



