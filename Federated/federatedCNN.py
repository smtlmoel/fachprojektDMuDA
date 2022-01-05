import math

import torch
import torchvision.datasets as datasets
from torch import nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from CNNNet import Net
from clientThread import ClientThread
import queue

import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def train(num_clients, epochs, communication_rounds, batch_size, experiment_name, aggregation="normal", layer_list=None, mask=None):
    if layer_list is None:
        layer_list = []
    # Move network to device GPU or CPU
    if torch.cuda.is_available():
        dev = "cuda:0"
        print("Device for training: cuda")
    else:
        dev = "cpu"
        print("Device for training: cpu")
    device = torch.device(dev)

    # Initialise Models
    global_network = torch.load("models/initial_weights.pth")
    global_network.to(device)
    networks = [torch.load("models/initial_weights.pth") for _ in range(num_clients)]
    for network in networks:
        network.to(device)

    # Load Cifar-10 Dataset
    cifar_train = datasets.CIFAR10(root='CIFAR_data/',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    cifar_test = datasets.CIFAR10(root='CIFAR_data/',
                                  train=False,
                                  transform=transforms.ToTensor(),
                                  download=True)

    train_data_split = torch.utils.data.random_split(cifar_train,
                                                     [int(cifar_train.data.shape[0] / num_clients) for _ in range(num_clients)],
                                                     generator=torch.Generator().manual_seed(42))

    train_loaders = [torch.utils.data.DataLoader(dataset=x,
                                                 batch_size=batch_size,
                                                 num_workers=1,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 drop_last=True) for x in train_data_split]

    # Initialize loss as CrossEntropyLoss
    crosslosses = [nn.CrossEntropyLoss() for _ in networks]
    # Initialize optimizers as SDG-optimizers
    optimizers = [torch.optim.Adam(network.parameters(), lr=0.0001, weight_decay=5e-4) for network in networks]

    # Loss
    federated_loss = [{'epoch_loss': [], 'batch_loss': []} for _ in range(num_clients)]

    # Federated training
    output_file = open(f"logs/output_{experiment_name}.txt", "w")
    output_file.write("Start federated training:\n")
    print("Start federated training:")
    loss_dict_queue = queue.Queue()

    for t in range(communication_rounds):
        output_file.write(f"Start communication round {t}:\n")
        print(f'Start communication round {t}: \n')

        threads = [ClientThread(idx, epochs,
                                train_loaders[idx],
                                networks[idx],
                                optimizers[idx],
                                crosslosses[idx],
                                loss_dict_queue,
                                output_file) for idx in range(num_clients)]
        if device == "cuda:0":
            for i, thread in enumerate(threads):
                thread.setName(f"ClientThread_{i}")

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()
                loss_dict = loss_dict_queue.get()
                idx = loss_dict.get('idx')
                federated_loss[idx].get('batch_loss').extend(loss_dict.get('batch_loss'))
                federated_loss[idx].get('epoch_loss').extend(loss_dict.get('epoch_loss'))
        else:  # device == "cpu"
            for i, thread in enumerate(threads):
                # result = ClientThread.local_learner(self, device, epochs, train_loaders[idx], networks[idx], optimizers[idx], crosslosses[idx])
                thread.setName(f"ClientThread_{i}")
                thread.start()
                thread.join()
                loss_dict = loss_dict_queue.get()
                idx = loss_dict.get('idx')
                federated_loss[idx].get('batch_loss').extend(loss_dict.get('batch_loss'))
                federated_loss[idx].get('epoch_loss').extend(loss_dict.get('epoch_loss'))

        # Load parameters from individual networks and average in global network
        output_file.write(f"Aggregation type is: {aggregation}.\n")
        print(f"Aggregation type is: {aggregation}.")
        if aggregation == "layers":
            global_dict = global_network.state_dict()
            for k in global_dict.keys():
                if k not in layer_list:
                    global_dict[k] = torch.stack([networks[i].state_dict()[k].float() for i in range(len(networks))], 0).mean(0)
            global_network.load_state_dict(global_dict)
            # Load new parameters to individual networks
            for network in networks:
                for k in global_dict.keys():
                    if k in layer_list:
                        global_dict[k] = network.state_dict()[k]
                network.load_state_dict(global_network.state_dict())

        elif aggregation == "mask":
            global_dict = global_network.state_dict()
            for k in global_dict.keys():
                global_dict[k] = torch.stack([networks[i].state_dict()[k].float() for i in range(len(networks))], 0).mean(0)
                global_dict[k] = global_dict[k] * mask[k]
            # Load new parameters to individual networks

            # TODO replace 0 with original client value

        else:  # aggregation == "normal"
            global_dict = global_network.state_dict()
            for k in global_dict.keys():
                global_dict[k] = torch.stack([networks[i].state_dict()[k].float() for i in range(len(networks))], 0).mean(0)
            global_network.load_state_dict(global_dict)
            # Load new parameters to individual networks
            for network in networks:
                network.load_state_dict(global_network.state_dict())

    # Load test data to cpu (gpu memory would overflow)
    test_loader = torch.utils.data.DataLoader(dataset=cifar_test,
                                              batch_size=len(cifar_test),
                                              shuffle=False)
    # Plotting
    plt.style.use(['seaborn-dark-palette', 'ggplot'])

    # Plotting batch loss
    batch_loss_length = 0
    for idx, loss_tuple in enumerate(federated_loss):
        batch_loss = loss_tuple.get('batch_loss')
        batch_loss_length = len(batch_loss)
        plt.plot([i for i in range(0, len(batch_loss))], batch_loss, label=f'Client {idx}')

    for communication_round in range(communication_rounds-1):
        plt.axvline(x=batch_loss_length/communication_rounds*(communication_round+1), color='black')

    communication_line = mlines.Line2D([0], [0], color='Black', label='Communication Rounds')

    plt.title('Federated clients: loss per batch')
    plt.xlabel('batch_id')
    plt.ylabel('batch_loss')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([communication_line])
    plt.legend(handles=handles)
    plt.savefig(f'figs/fedClients_lossPerBatch_{experiment_name}.png')
    plt.show()

    # Plotting epoch loss
    epoch_loss_length = 0
    for idx, loss_tuple in enumerate(federated_loss):
        epoch_loss = loss_tuple.get('epoch_loss')
        epoch_loss_length = len(epoch_loss)
        plt.plot([i for i in range(0, len(epoch_loss))], epoch_loss, label=f'Client {idx}')

    for communication_round in range(communication_rounds-1):
        plt.axvline(x=epoch_loss_length/communication_rounds*(communication_round+1)-0.5, color='black')

    plt.title('Loss per epoch')
    plt.xlabel('epoch_id')
    plt.ylabel('epoch_loss')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([communication_line])
    plt.legend(handles=handles)
    plt.savefig(f'figs/epochLoss_{experiment_name}.png')
    plt.show()

    # Test global network
    global_network.eval()
    for x_test, y_test in test_loader:
        global_network.to('cpu')
        prediction = global_network(x_test)
        correct_prediction = (torch.max(prediction.data, dim=1)[1] == y_test.data)
        accuracy = correct_prediction.float().mean().item()
        s = 'Global Accuracy: {:2.2f}%'.format(accuracy * 100)
        output_file.write(s+"\n")
        print(s)

    output_file.close()


def main():
    train(num_clients=4, epochs=10, communication_rounds=5, batch_size=64, experiment_name="federatedCNNMain")


if __name__ == '__main__':
    main()
