import torchvision
import torch
import os


def find_normalization_constants():
    """Find constants needed to normalize training data"""

    # transformations to be applied to training data
    trainDataTransforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.RandomResizedCrop(size=48),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ]
    )

    # load training images (in PIL form)
    train_data = torchvision.datasets.ImageFolder(root=os.path.abspath('data/images/training'),
                                                  transform=trainDataTransforms)

    imgs = torch.stack([img for img, _ in train_data], dim=3)

    # calculate mean and standard deviation for imgs tensor containing all training images
    training_set_mean = torch.mean(imgs).item()
    training_set_std = torch.std(imgs).item()

    return training_set_mean, training_set_std


def load_data():
    """Load images from data directory, apply transformation, convert to tensor and normalize"""

    training_set_mean, training_set_std = find_normalization_constants()

    trainDataTransforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.RandomResizedCrop(size=48),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([training_set_mean], [training_set_std])
        ]
    )

    validationTestDataTransforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.Resize((48, 48)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([training_set_mean], [training_set_std])
        ]
    )

    train_data = torchvision.datasets.ImageFolder(root=os.path.abspath('data/images/training'),
                                                  transform=trainDataTransforms)
    validation_data = torchvision.datasets.ImageFolder(root=os.path.abspath('data/images/validation'),
                                                       transform=validationTestDataTransforms)
    test_data = torchvision.datasets.ImageFolder(root=os.path.abspath('data/images/test'),
                                                 transform=validationTestDataTransforms)

    return train_data, validation_data, test_data, validationTestDataTransforms


def get_data_loaders():
    """Define DataLoaders for training, validation, and testing data"""

    BATCH_SIZE = 64

    train_data, validation_data, test_data, validationTestDataTransforms = load_data()

    trainDataLoader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
    validationDataLoader = torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE, num_workers=2,
                                                       shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)

    data_loaders = {'Train': trainDataLoader, 'Validation': validationDataLoader, 'Test': testDataLoader}

    return data_loaders, validationTestDataTransforms
