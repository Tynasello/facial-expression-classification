## Facial Expression Classification

The goal of this project was to use deep learning to classify images of people based on their emotions.

### The Dataset

The dataset for this project comes from the [Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) on Kaggle. It contains images of faces categorized under seven emotions: anger, disgust, fear, happy, sad, surprise, and neutral. The dataset is composed of training, validation and test data containing 28709, 3589, and 3589 samples respectively.

### About the Project

This project involved the use of transfer learning and a CNN (convolutional neural network). Data augmentation was performed to the data set to increase model performance while reducing the likelihood of overfitting.

Initially a custom model was developed. This model architecture consisted of 6 convolutional layers with ReLU activation functions, and max pooling, followed by 3 linear layers. By first creating my own model, I gained a better understanding of convolutional layers and feature mapping as well as the effect of different kernel sizes, padding and stride values, and pooling layers. This model achieved ~54% accuracy on the validation set.

I then employed transfer learning using the ResNet-50 pre-trained model. Upon adding my own classification head, this architecture achieved ~63% accuracy on the test data. After visualizing images with wrongly predicted labels, it was clear that a percentage of them were mislabeled or indeterminate to the human eye.

Throughout this project, the training of models was done on Google Colab using a GPU.

To load the model wth trained parameters:
`emotion_model.load_state_dict(torch.load('model_parameters/facial_expression_model.pth'))`

---

A precursor to this project was my MNIST handwritten digit classifier. This introductory [project](https://github.com/Tynasello/mnist-digit-classifier) provided me with experience in PyTorch and allowed me to apply what I had learned about deep learning and neural networks up to that point.

Technologies and tools used in this project: Python, Jupyter Notebooks, Google Colab, PyTorch, NumPy, Pandas, Matplotlib.

### Resources

Below are links to resources that provided me with a better understanding of various concepts and aided in the completion of this project.

Model Architecture

https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

CNN fundamentals

https://www.analyticsvidhya.com/blog/2018/12/guide-convolutional-neural-network-cnn/?utm_source=blog&utm_medium=building-image-classification-models-cnn-pytorch

Data augmentation

https://medium.com/swlh/how-data-augmentation-improves-your-cnn-performance-an-experiment-in-pytorch-and-torchvision-e5fb36d038fb

Transfer Learning

https://stackabuse.com/image-classification-with-transfer-learning-and-pytorch/
https://towardsdatascience.com/transfer-learning-with-convolutional-neural-networks-in-pytorch-dd09190245ce
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

Learning rate schedulers and adaptive learning rates

https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1

Confusion Matrix

https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7