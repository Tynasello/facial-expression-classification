import torch
import os


# Mapping emotion ids to meaning
EMOTIONS_TO_ID = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}


def to_t(tensor):
    """"Convert tensor to GPU if available"""

    return tensor.to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))


def to_cpu(tensor):
    """"Convert tensor to CPU"""

    return tensor.to(torch.device('cpu'))


def createFolders():
    """Create needed folder in project directory. Used for notebooks."""

    for folder_name in ["models", "data"]:
        if not os.path.exists(os.path.abspath(folder_name)):
            os.mkdir(os.path.abspath(folder_name))

    for folder_name in ["images", "images/training", "images/validation", "images/test",
                        "my_emotions", "outputs", "outputs/raytune_result"]:
        if not os.path.exists(os.path.abspath('data/'+folder_name)):
            os.mkdir(os.path.abspath('data/'+folder_name))

    for emotion in EMOTIONS_TO_ID:
        for folder_name in ["images/training/", "images/validation/", "images/test/"]:
            if not os.path.exists(os.path.abspath('data/'+folder_name + str(emotion))):
                os.mkdir(os.path.abspath('data/'+folder_name + str(emotion)))


def validate_imgs_normalized(train_data):
    """Validate that images in training data have been normalized. ie their mean is 0 and standard deviation is 1"""

    imgs = torch.stack([img for img, _ in train_data], dim=3)
    print("mean: " + str(imgs.mean().item()))
    print("standard deviation: " + str(imgs.std().item()))


def save_model(model):
    """Save model parameters"""
    torch.save(model.state_dict(), os.path.abspath("models/facial_expression_model.pth"))

