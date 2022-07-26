import torch
import numpy as np
import copy
import os
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from src.helpers import to_t
from src.models import models


def train_model(config, data_loaders, model_id, num_epochs=30, tuning=False):
    """Train model"""


    # get desired model, loss function and optimization function
    if model_id == "custom":
        model, loss_fn, optimizer = models.get_custom_model()
    else:
        model, loss_fn, optimizer = models.get_pretrained_model(config["l1"], config["l2"], config["lr"])

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
                if not tuning and epoch_accuracy > highest_accuracy:
                    highest_accuracy = epoch_accuracy
                    best_model_params = copy.deepcopy(model.state_dict())

            # print performance for current epoch
            print(f"{phase}")
            print(f"Loss: {epoch_loss:.4f}   Acc: {epoch_accuracy:.4f}")

        # report results to raytune
        if tuning:
            report_raytune(epoch, model, optimizer, validation_losses[-1], validation_accuracies[-1])

        print()
        print()

    # load best performing model parameters
    model.load_state_dict(best_model_params)
    return model, train_losses, train_accuracies, validation_losses, validation_accuracies


def report_raytune(epoch, model, optimizer, validation_loss, validation_accuracy):
    """Save raytune checkpoint with hyperparameters and report to raytune with performance results"""

    with tune.checkpoint_dir(epoch) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, 'checkpoint')
        torch.save((model.state_dict(), optimizer.state_dict()), path)
    tune.report(loss=validation_loss, accuracy=validation_accuracy)


def run_hyperparameter_search(data_loaders, model_id, num_epochs):
    """Attempt to find the best hyperparameters for desired model"""

    # tuning the following hyperparameters: learning rate, linear layer 1 num ouput features, linear layer 2 num ouput features.
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(5, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(5, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
    }

    # monitor loss metric and stop bad performing searches
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        # max num of epochs to train with given hyperparameters
        max_t=10,
        # stop bad performing search after 3 epochs
        grace_period=3,
        reduction_factor=2)

    # report metrics in terminal
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])

    # start hyperparameter search
    result = tune.run(
        tune.with_parameters(train_model, data_loaders=data_loaders, model_id=model_id, num_epochs=num_epochs,
                             tuning=True),
        resources_per_trial={"cpu": 1, "gpu": 1},
        config=config,
        # try 20 different hyperparameter combinations
        num_samples=20,
        scheduler=scheduler,
        # save results search trials
        local_dir=os.path.abspath('data/outputs/raytune_result'),
        # save one checkpoint for each search trial
        keep_checkpoints_num=1,
        # save checkpoint based on minimum validation loss
        checkpoint_score_attr='min-validation_loss',
        progress_reporter=reporter
    )

    # Extract the best trial run from the search.
    best_trial = result.get_best_trial(
        'loss', 'min', 'last'
    )

    # return best hyperparameters and best trial validation loss and accuracy
    return best_trial.config, best_trial.last_result['loss'], best_trial.last_result['accuracy']
