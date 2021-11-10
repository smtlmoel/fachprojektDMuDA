import torch
import torchvision.datasets as dsets
from torch import nn as nn
import torch.utils.data
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


class SmallCNN(torch.nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        '''
        ConvLayer 0
        ImageIn (28x28x1)
        -> Conv2d (24x24x32)
        -> ReLu
        -> MaxPool (12x12x32)
        '''
        self.convLayer0 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Dropout(0.4)
        )
        '''
        ConvLayer 1
        ImageIn (12x12x32)
        -> Conv2d (8x8x64)
        -> ReLu
        -> MaxPool (4x4x64)
        '''
        self.convLayer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Dropout(0.4)
        )
        '''
        Input (4x4x64)
        -> Flatten
        -> Output (1024)
        '''
        self.flat = nn.Flatten()
        '''
        Input (1024)
        -> Relu
        -> Output (128)
        '''
        self.fcLayer0 = nn.Sequential(
            nn.Linear(4 * 4 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        '''
        Input (128)
        -> Relu
        -> Output (10)
        '''
        self.fcLayer1 = nn.Sequential(
            nn.Linear(128, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.convLayer0(x)
        x = self.convLayer1(x)
        x = self.flat(x)
        x = self.fcLayer0(x)
        out = self.fcLayer1(x)
        return out


class LargeCNN(torch.nn.Module):
    def __init__(self):
        super(LargeCNN, self).__init__()
        '''
        ConvLayer 0
        ImageIn (28x28x1)
        -> Conv2d (28x28x32)
        -> ReLu
        -> MaxPool (14x14x32)
        '''
        self.convLayer0 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Dropout(0.4)
        )
        '''
        ConvLayer 1
        ImageIn (14x14x32)
        -> Conv2d (14x14x64)
        -> ReLu
        -> MaxPool (7x7x64)
        '''
        self.convLayer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Dropout(0.4)
        )
        '''
        ConvLayer 2
        ImageIn (7x7x64)
        -> Conv2d (7x7x128)
        -> ReLu
        -> MaxPool (3x3x128)
        '''
        self.convLayer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), 2),
            nn.Dropout(0.4)
        )
        '''
        ConvLayer 3
        ImageIn (3x3x128)
        -> Conv2d (3x3x256)
        -> ReLu
        -> MaxPool (1x1x256)
        '''
        self.convLayer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), 1),
            nn.Dropout(0.4)
        )
        '''
        Input (256)
        -> Relu
        -> Output (512)
        '''
        self.fcLayer0 = nn.Sequential(
            nn.Linear(256, 512, bias=True),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        '''
        Input (512)
        -> Relu
        -> Output (512)
        '''
        self.fcLayer1 = nn.Sequential(
            nn.Linear(512, 512, bias=True),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        '''
        Input (512)
        -> Relu
        -> Output (10)
        '''
        self.fcLayer2 = nn.Sequential(
            nn.Linear(512, 10, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.convLayer0(x)
        x = self.convLayer1(x)
        x = self.convLayer2(x)
        x = self.convLayer3(x)
        x = x.view(x.size(0), -1)
        x = self.fcLayer0(x)
        x = self.fcLayer1(x)
        out = self.fcLayer2(x)
        return out


def mnist(epoch: int, batch_size: int, network: torch.nn.Module):
    # Move network to device GPU or CPU
    if torch.cuda.is_available():
        dev = "cuda:0"
        print("Device for training: cuda")
    else:
        dev = "cpu"
        print("Device for training: cpu")
    device = torch.device(dev)
    network.to(device)

    # Load MNIST dataset
    mnist_train = dsets.MNIST(root='MNIST_data/',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)

    mnist_test = dsets.MNIST(root='MNIST_data/',
                             train=False,
                             transform=transforms.ToTensor(),
                             download=True)

    # Initialize dataset loader
    train_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                               batch_size=batch_size,
                                               shuffle=True)

    #  Initialize loss as CrossEntropyLoss
    crossloss = nn.CrossEntropyLoss()

    # Initialize optimizer as SDG-optimizer
    optimizer = torch.optim.SGD(network.parameters(), lr=0.03)

    # Initialize variable to trace loss
    batch_loss = []
    epoch_loss = []

    # Start training
    for epoch in range(epoch):
        for i, (batch_X, batch_Y) in enumerate(train_loader):
            x = batch_X.to(device)
            y = batch_Y.to(device)

            # Set gradients to zero
            optimizer.zero_grad()

            # Train network
            output = network.forward(x)
            # Calculate loss with CrossEntropyLoss
            loss = crossloss(output, y)

            # Back-propagate loss
            loss.backward()
            optimizer.step()

            # Append loss to all_loss for tracking
            with torch.no_grad():
                batch_loss.append(loss.cpu().detach().numpy())

        # print progress
        if epoch % 1 == 0:
            s = f'Epoch: {epoch + 1} completed. Current loss: {batch_loss[-1]} '
            print(s)
            # Print results to log.txt
            with open(r"log.txt", "a") as f:
                f.write(s.__add__('\n'))
            f.close()

        epoch_loss.append(batch_loss[-1])

    # Style for plots
    plt.style.use(['seaborn-dark-palette', 'ggplot'])

    # plot the loss over all training batches
    plt.title('batch loss')
    plt.plot([i for i in range(0, len(batch_loss))], batch_loss)
    plt.xlabel('batch_id')
    plt.ylabel('batch_loss')
    plt.show()

    # plot the loss over all epochs
    plt.title('epoch loss')
    plt.plot([i+1 for i in range(0, len(epoch_loss))], epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('epoch_loss')
    plt.show()

    # Load test data to cpu (gpu memory would overflow)
    x_test = mnist_test.data.view(len(mnist_test), 1, 28, 28).float()
    y_test = mnist_test.targets
    x = x_test.to("cpu")
    y = y_test.to("cpu")
    network.to("cpu")

    # Calculate accuracy and print it
    prediction = network(x)
    correct_prediction = (torch.max(prediction.data, dim=1)[1] == y.data)
    accuracy = correct_prediction.float().mean().item()
    s = '\n Accuracy: {:2.2f}%'.format(accuracy * 100)
    print(s)
    # Print result to log.txt
    with open(r"log.txt", "a") as f:
        f.write(s)
    f.close()


def main(args):
    print("SmallCNN:")
    mnist(10, 64, SmallCNN())
    print("LargeCNN:")
    mnist(10, 64, LargeCNN())


if __name__ == '__main__':
    main("test")