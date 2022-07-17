import torch
import torchvision


class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.convolutionalLayers = torch.nn.Sequential(

            torch.nn.Conv2d(3, 32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.ReLU(),

            torch.nn.Flatten()
        )

        self.linearLayers = torch.nn.Sequential(
            torch.nn.Linear(800, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 7),
        )

        self.to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    def forward(self, x):
        # Convolutional layers
        out = self.convolutionalLayers(x)

        # Linear layers
        out = self.linearLayers(out)

        return out


def get_custom_model():
    custom_facial_expression_model = CustomModel()
    custom_model_loss_fn = torch.nn.CrossEntropyLoss()
    custom_model_optimizer = torch.optim.Adam(custom_facial_expression_model.parameters())

    return custom_facial_expression_model, custom_model_loss_fn, custom_model_optimizer


def get_pretrained_model(l1=128, l2=128, lr=7e-3):
    # pretrained_facial_expression_model = torchvision.models.resnet50(pretrained=True)
    pretrained_facial_expression_model = torchvision.models.resnet50(weights="ResNet50_Weights.DEFAULT")

    num_ftrs = pretrained_facial_expression_model.fc.in_features

    # classification head
    pretrained_facial_expression_model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, l1),
        torch.nn.ReLU(),
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, 7),
    )

    # model to GPU if available
    pretrained_facial_expression_model.to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    # loss function - cross entropy loss
    # optimizer - Stochastic gradient descent
    model_loss_fn = torch.nn.CrossEntropyLoss()
    model_optimizer = torch.optim.SGD(pretrained_facial_expression_model.parameters(), lr=lr, momentum=0.9)

    return pretrained_facial_expression_model, model_loss_fn, model_optimizer
