#!/usr/bin/env python3

from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
from torchviz import make_dot
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import sklearn.model_selection
import matplotlib.pyplot as plt
import numpy as np
import time

import image_loader


def disp_image(image):
    # need to reorder the tensor dimensions to work properly with imshow
    reshaped = np.asarray(image).transpose(1, 2, 0)
    plt.imshow(reshaped)
    plt.axis("off")
    plt.show()


image_h = 240
image_w = 256

# Image format is: list of colors, each element is (list of rows, each element is (list of pixels))
# i.e. the middle dimension is height


class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()

        self.activation_func = torch.nn.ReLU()

        num_kernels = 16
        self.conv1 = nn.Conv2d(3, num_kernels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_out_size = num_kernels * (image_h // 2) * (image_w // 2)
        fc1_size = 64
        self.fc1 = nn.Linear(self.pool_out_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, 2)  # 2 output classes (uncursed, cursed)

    def forward(self, x):

        conv1_out = self.conv1(x)
        pool_out = self.pool(conv1_out)
        act1_out = self.activation_func(pool_out)
        fc1_in = pool_out.view(-1, self.pool_out_size)
        fc1_out = self.fc1(fc1_in)
        act2_out = self.activation_func(fc1_out)
        fc2_out = self.fc2(act2_out)

        return fc2_out

    def get_loss(self, learning_rate):

        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        return loss, optimizer


def make_loader(dataset, batch_size):
    sampler = SubsetRandomSampler(np.arange(len(dataset)))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=2
    )
    return loader


def train_model(net, n_epochs, learning_rate, train_loader, test_loader):
    """ Train a the specified network.

        Outputs a tuple with the following four elements
        train_hist_x: the x-values (batch number) that the training set was 
            evaluated on.
        train_loss_hist: the loss values for the training set corresponding to
            the batch numbers returned in train_hist_x
        test_hist_x: the x-values (batch number) that the test set was 
            evaluated on.
        test_loss_hist: the loss values for the test set corresponding to
            the batch numbers returned in test_hist_x
    """
    loss, optimizer = net.get_loss(learning_rate)
    # Define some parameters to keep track of metrics
    print_every = 20
    idx = 0
    train_hist_x = []
    train_loss_hist = []
    test_hist_x = []
    test_loss_hist = []

    training_start_time = time.time()
    # Loop for n_epochs
    for epoch in range(n_epochs):
        running_loss = 0.0
        start_time = time.time()

        for i, data in enumerate(train_loader):

            # Get inputs in right form
            inputs, labels = data
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

            # In Pytorch, We need to always remember to set the optimizer gradients to 0 before we recompute the new gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)

            # Compute the loss and find the loss with respect to each parameter of the model
            loss_size = loss(outputs, labels.long())
            loss_size.backward()

            # Change each parameter with respect to the recently computed loss.
            optimizer.step()

            # Update statistics
            running_loss += loss_size.data.item()

            # Print every 20th batch of an epoch
            if (i % print_every) == print_every - 1:
                print(
                    "Epoch {}, Iteration {}\t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch + 1,
                        i + 1,
                        running_loss / print_every,
                        time.time() - start_time,
                    )
                )
                # Reset running loss and time
                train_loss_hist.append(running_loss / print_every)
                train_hist_x.append(idx)
                running_loss = 0.0
                start_time = time.time()
            idx += 1

        # At the end of the epoch, do a pass on the test set
        total_test_loss = 0
        for inputs, labels in test_loader:

            # Wrap tensors in Variables
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

            # Forward pass
            test_outputs = net(inputs)
            test_loss_size = loss(test_outputs, labels.long())
            total_test_loss += test_loss_size.data.item()
        test_loss_hist.append(total_test_loss / len(test_loader))
        test_hist_x.append(idx)
        print("Validation loss = {:.2f}".format(total_test_loss / len(test_loader)))

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    return train_hist_x, train_loss_hist, test_hist_x, test_loss_hist


if __name__ == "__main__":

    dataset = image_loader.load_images("dataset1", (image_h, image_w))

    dataset_train, dataset_test = sklearn.model_selection.train_test_split(
        dataset, test_size=0.25
    )

    train_loader = make_loader(dataset_train, 32)
    test_loader = make_loader(dataset_test, 32)

    device = "cpu"  # Change to "cuda" if on a machine with cuda
    net = cnn()
    net.to(device)

    train_hist_x, train_loss_hist, test_hist_x, test_loss_hist = train_model(
        net, 10, 1e-2, train_loader, test_loader
    )
