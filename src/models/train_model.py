import torch
import copy

from src.helpers import to_t


def train_model(model, data_loaders, loss_fn, optimizer, num_epochs):
    """Train model"""

    train_losses, train_accuracies, validation_losses, validation_accuracies = [], [], [], []

    best_model_params = copy.deepcopy(model.state_dict())
    highest_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}')
        print('-' * 7)

        for phase in ["Train", "Validation"]:
            # set models mode
            if phase == "Train":
                model.train()
            else:
                model.eval()

            # for calculating model loss and accuracy for current epoch
            total_loss = 0
            num_predictions = 0
            num_correct_predictions = 0

            for images, labels in data_loaders[phase]:
                # convert tensors to gpu if available
                images = to_t(images)
                labels = to_t(labels)

                # don't accumulate gradients
                optimizer.zero_grad()

                # only calculate gradients for train data
                with torch.set_grad_enabled(phase == "Train"):
                    # complete forward pass with images in current batch
                    outputs = model.forward(images)
                    # calculate predicted emotions for current batch
                    predictions = torch.argmax(outputs.data, 1)
                    # calculate loss for current batch
                    loss = loss_fn(outputs, labels)

                    # only compute gradients and step parameters for training data
                    if phase == "Train":
                        # compute gradients using back propagation
                        loss.backward()
                        # step model parameters based on calculated gradients
                        optimizer.step()

                # update model loss for current epoch
                total_loss += loss.item() * labels.size(0)
                # update total num of predictions and num correct predictions for current batch
                num_predictions += labels.size(0)
                num_correct_predictions += (predictions == labels).sum()

            # calculate model loss and accuracy for current epoch
            epoch_loss = total_loss / num_predictions
            epoch_accuracy = (100 * num_correct_predictions / num_predictions).item()

            if phase == "Train":
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_accuracy)
            else:
                validation_losses.append(epoch_loss)
                validation_accuracies.append(epoch_accuracy)
                # if our model is performing at its best (for validation data), update best parameters and highest accuracy
                if epoch_accuracy > highest_accuracy:
                    highest_accuracy = epoch_accuracy
                    best_model_params = copy.deepcopy(model.state_dict())

            # print performance for current epoch
            print(f"{phase}")
            print(f"Loss: {epoch_loss:.4f}   Acc: {epoch_accuracy:.4f}")

        print()
        print()

    # load best performing model parameters
    model.load_state_dict(best_model_params)
    return model, train_losses, train_accuracies, validation_losses, validation_accuracies

