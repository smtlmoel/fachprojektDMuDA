import torch
import torchvision.datasets as datasets
from torch import nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from clientThread import ClientThread
import queue

import matplotlib.pyplot as plt


def train(epochs, batch_size, experiment_name):
    # Move network to device GPU or CPU
    if torch.cuda.is_available():
        dev = "cuda:0"
        print("Device for training: cuda")
    else:
        dev = "cpu"
        print("Device for training: cpu")
    device = torch.device(dev)

    # Initialise Model
    central_network = torch.load("models/initial_weights.pth")
    central_network.to(device)

    # Load Cifar-10 Dataset
    cifar_train = datasets.CIFAR10(root='CIFAR_data/',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    cifar_test = datasets.CIFAR10(root='CIFAR_data/',
                                  train=False,
                                  transform=transforms.ToTensor(),
                                  download=True)

    central_loader = torch.utils.data.DataLoader(dataset=cifar_train,
                                                 batch_size=batch_size,
                                                 num_workers=4,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 drop_last=True)

    # Initialize loss as CrossEntropyLoss
    central_crossloss = nn.CrossEntropyLoss()

    # Initialize optimizer as Adam-optimizer
    central_optimizer = torch.optim.Adam(central_network.parameters(), lr=0.0001, weight_decay=5e-4)

    # Loss
    central_loss = {'epoch_loss': [], 'batch_loss': []}

    output_file = open(f"logs/output_{experiment_name}.txt", "w")
    loss_dict_queue = queue.Queue()

    # Central training
    output_file.write("Start central training:\n")
    print("Start central training:")
    central_thread = ClientThread("Central",
                                  epochs,
                                  central_loader,
                                  central_network,
                                  central_optimizer,
                                  central_crossloss,
                                  loss_dict_queue,
                                  output_file)
    central_thread.start()
    central_thread.join()
    loss_dict = loss_dict_queue.get()
    central_loss.get('batch_loss').extend(loss_dict.get('batch_loss'))
    central_loss.get('epoch_loss').extend(loss_dict.get('epoch_loss'))

    # Load test data to cpu (gpu memory would overflow)
    test_loader = torch.utils.data.DataLoader(dataset=cifar_test,
                                              batch_size=len(cifar_test),
                                              shuffle=False)
    # Plotting
    plt.style.use(['seaborn-dark-palette', 'ggplot'])

    # Plotting batch loss
    plt.title('Central Network: loss per batch')
    plt.xlabel('batch_id')
    plt.ylabel('batch_loss')
    plt.plot([i for i in range(0, len(central_loss.get('batch_loss')))], central_loss.get('batch_loss'),
             label='Central Network')
    plt.legend()
    plt.savefig(f'figs/centralNetwork_lossPerBatch_{experiment_name}.png')
    plt.show()

    # Plotting epoch loss
    plt.plot([i for i in range(0, len(central_loss.get('epoch_loss')))], central_loss.get('epoch_loss'),
             label='Central Network')
    plt.title('Loss per epoch')
    plt.xlabel('epoch_id')
    plt.ylabel('epoch_loss')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles)
    plt.savefig(f'figs/epochLoss_{experiment_name}.png')
    plt.show()

    # Test central network
    central_network.eval()
    for x_test, y_test in test_loader:
        central_network.to('cpu')
        prediction = central_network(x_test)
        correct_prediction = (torch.max(prediction.data, dim=1)[1] == y_test.data)
        accuracy = correct_prediction.float().mean().item()
        s = 'Central Accuracy: {:2.2f}%'.format(accuracy * 100)
        output_file.write(s+"\n")
        print(s)

    output_file.close()


def main():
    train(epochs=10, batch_size=64, experiment_name="centralCNNMain")


if __name__ == '__main__':
    main()
