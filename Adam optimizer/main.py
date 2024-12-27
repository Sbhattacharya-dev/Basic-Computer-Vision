from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from matplotlib import pyplot as plt

# Set the seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load the training and test data
train_data = MNIST(root='data', train=True, download=True)
test_data = MNIST(root='data', train=False, download=True)
# Extract the images and labels
train_images, train_labels = train_data.data, train_data.targets
test_images, test_labels = test_data.data, test_data.targets
# Print the shape of the data
print('Train images shape:', train_images.shape, 'Train labels shape:', train_labels.shape)
print('Test images shape:', test_images.shape, 'Test labels shape:', test_labels.shape)


# Task 0: Do some data visualization and preprocessing here

# (1) Randomly pick some samples and view individual images and corresponding labels
import random

def view_samples(images, labels, num_samples=10):
    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        index = random.randint(0, len(images) - 1)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[index], cmap='gray')
        plt.title(f'Label: {labels[index].item()}')
        plt.axis('off')
    plt.savefig('sample_images.png')
    plt.clf()

view_samples(train_images, train_labels)

# (2) Analyze the distribution of digits in the dataset
def digit_distribution(labels):
    plt.figure(figsize=(10, 5))
    plt.hist(labels.numpy(), bins=10, edgecolor='k', alpha=0.7)
    plt.xlabel('Digit')
    plt.ylabel('Frequency')
    plt.title('Distribution of Digits in MNIST Dataset')
    plt.savefig('label_distribution.png')
    plt.clf()

digit_distribution(train_labels)

# (3) Generate statistical summaries of the dataset
def generate_statistical_summary(images):
    images = images.numpy()
    mean = np.mean(images)
    std = np.std(images)
    print(f'Mean: {mean:.4f}, Standard Deviation: {std:.4f}')

generate_statistical_summary(train_images)

# Task 1: define the model
# Task 1.1: Implement a Multi-Layer Perceptron (MLP) with 2 hidden layers

class MLP(nn.Module):
    def __init__(self, num_of_neurons_in_hidden_layer=256, num_of_classes=10):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(28*28, num_of_neurons_in_hidden_layer)
        self.hidden_layer1 = nn.Linear(num_of_neurons_in_hidden_layer, num_of_neurons_in_hidden_layer) # First hidden layer
        self.hidden_layer2 = nn.Linear(num_of_neurons_in_hidden_layer, num_of_neurons_in_hidden_layer) #Second hidden layer
        self.output_layer = nn.Linear(num_of_neurons_in_hidden_layer, num_of_classes) # Output layer

    def forward(self, x):
        assert len(x.shape) == 4 and x.shape[1:] == (1, 28, 28), 'Input shape should be (batch_size, 1, 28, 28)'
        x = x.view(-1, 28*28)  # Flatten the input
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        x = self.output_layer(x)
        return x



class CNN(nn.Module):
    def __init__(self, num_of_classes=10):
        super(CNN, self).__init__()
        # Define the layers of the CNN
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # First convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Second convolutional layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max-pooling layer
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, num_of_classes)  # Output layer

    def forward(self, x):
        assert len(x.shape) == 4 and x.shape[1:] == (1, 28, 28), 'Input shape should be (batch_size, 1, 28, 28)'
        x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
        x = self.pool(F.relu(self.conv2(x)))  # Apply second convolution, ReLU activation, and max-pooling
        x = x.view(x.size(0), -1)  # Flatten the tensor while keeping the batch size
        x = F.relu(self.fc1(x))  # Apply first fully connected layer and ReLU activation
        x = self.fc2(x)  # Apply output layer
        return x


class LeNet5(nn.Module):
    def __init__(self, num_of_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # First convolutional layer
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # Second convolutional layer
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # Average pooling layer
        self.fc1 = nn.Linear(16*4*4, 120)  # First fully connected layer
        self.fc2 = nn.Linear(120, 84)  # Second fully connected layer
        self.fc3 = nn.Linear(84, num_of_classes)  # Output layer

    def forward(self, x):
        assert len(x.shape) == 4 and x.shape[1:] == (1, 28, 28), 'Input shape should be (batch_size, 1, 28, 28)'
        x = self.pool(F.relu(self.conv1(x)))  # Apply first convolution, ReLU, and average pooling
        # print(f'After conv1 and pool: {x.shape}')
        x = self.pool(F.relu(self.conv2(x)))  # Apply second convolution, ReLU, and average pooling
        # print(f'After conv2 and pool: {x.shape}')
        x = x.view(-1, 16*4*4)  # Flatten the output of the pool layer
        # print(f'After flattening: {x.shape}')
        x = F.relu(self.fc1(x))  # Apply first fully connected layer and ReLU
        x = F.relu(self.fc2(x))  # Apply second fully connected layer and ReLU
        x = self.fc3(x)  # Apply output layer
        return x


# Define the models again to include debugging prints
models = {
    'MLP': MLP(),
    'CNN': CNN(),
    'LeNet5': LeNet5()
}

# Define training parameters
epochs = 3
batch_size = 64
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Task 2: implement the CrossEntropyLoss here and compare with the official torch API (opt. let's try L2 loss?)

# Task 2: implement the CrossEntropyLoss here and compare with L2 loss

# Define the CrossEntropyLoss and L2 loss
criterion_ce = nn.CrossEntropyLoss()

from torch.utils.data import DataLoader, TensorDataset
def l2_loss(outputs, labels):
    labels_one_hot = F.one_hot(labels, num_classes=10).float()
    outputs = F.softmax(outputs, dim=1)  # Apply softmax to the outputs
    return torch.mean((outputs - labels_one_hot) ** 2)

# Create DataLoader for training and test datasets
train_loader = DataLoader(TensorDataset(train_images.unsqueeze(1).float(), train_labels), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_images.unsqueeze(1).float(), test_labels), batch_size=batch_size, shuffle=False)

accuracy = []
losses = ['CrossEntropy', 'L2']
for loss_type in losses:
    for model_type, model in models.items():
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # try different optimizers?
        #Using another optimizer SGD
        #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print(f'Training {model_type} model with {loss_type} loss on {device} using {epochs} epochs, {batch_size} batch size and {learning_rate} learning rate')
        # train the model
        cur_accuracy = []
        for epoch in tqdm(range(epochs)):
            print('Epoch:', epoch)
            for i in range(0, len(train_images), batch_size):
                # Extract the batch
                batch_images = train_images[i:i+batch_size].to(device)
                batch_labels = train_labels[i:i+batch_size].to(device)
                # Forward pass
                outputs = model(batch_images.unsqueeze(1).float())
                if loss_type == 'CrossEntropy':
                    loss = criterion_ce(outputs, batch_labels)
                elif loss_type == 'L2':
                    loss = l2_loss(outputs, batch_labels)
                else:
                    raise NotImplementedError('Please Implement the CrossEntropyLoss and L2 loss in Task 2')
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Test the model
            predictions = model(test_images.unsqueeze(1).float().to(device))
            predictions = torch.argmax(predictions, dim=1)
            # Calculate the accuracy
            cur_accuracy.append(np.mean(predictions.cpu().numpy() == test_labels.cpu().numpy()))
            print('Accuracy:', cur_accuracy[-1])

        # Plot the current accuracy curve
        plt.plot(range(1, epochs+1), cur_accuracy, label=f'{model_type} with {loss_type}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'{model_type} Accuracy with {loss_type}')
        plt.legend()
        plt.savefig(f'{model_type}_Accuracy_{loss_type}.png')
        plt.clf()
        accuracy.append((loss_type, model_type, cur_accuracy[-1]))

# Plot the accuracy comparison
loss_types, model_types, acc_values = zip(*accuracy)
plt.bar(range(len(acc_values)), acc_values)
plt.xticks(range(len(acc_values)), [f'{lt}-{mt}' for lt, mt in zip(loss_types, model_types)], rotation=45)
for i, acc in enumerate(acc_values):
    plt.text(i, acc, f'{acc:.4f}', ha='center', va='bottom')
plt.xlabel('Model and Loss Type')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Different Models and Loss Functions')
plt.savefig('Accuracy_Comparison.png')
plt.show()
