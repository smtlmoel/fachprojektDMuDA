import torch
import torchvision.datasets as datasets
from torch import nn as nn
import torch.utils.data
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# TODO Multi-thread clients
# TODO Track communication size


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        '''
        ConvLayer 0
        ImageIn (32x32x3)
        -> Conv2d (28x28x64)
        -> ReLu
        -> MaxPool (14x14x32)
        -> Dropout 40%
        '''
        self.convLayer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Dropout(0.4)
        )
        '''
        ImageIn (14x14x64)
        -> Conv2d (10x10x128)
        -> ReLu
        -> MaxPool (5x5x128)
        -> Dropout 40%
        '''
        self.convLayer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Dropout(0.4)
        )
        '''
        ImageIn (5x5x128)
        -> Flatten
        -> Output (3200)
        '''
        self.flat = nn.Flatten()
        '''
        Input (3200)
        -> Relu
        -> Dropout 40%
        -> Output (512)
        '''
        self.fcLayer0 = nn.Sequential(
            nn.Linear(3200, 512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        '''
        Input (256)
        -> RelU
        -> Dropout 40%
        -> Output (10)
        '''
        self.fcLayer1 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout()
        )
        '''
        Input (128)
        -> ReLu
        -> Output (10)
        '''
        self.fcLayer2 = nn.Sequential(
            nn.Linear(128, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.convLayer0(x)
        x = self.convLayer1(x)
        x = self.flat(x)
        x = self.fcLayer0(x)
        x = self.fcLayer1(x)
        out = self.fcLayer2(x)
        return out


def train(num_clients, epochs, communication_rounds, batch_size):
    # Move network to device GPU or CPU
    if torch.cuda.is_available():
        dev = "cuda:0"
        print("Device for training: cuda")
    else:
        dev = "cpu"
        print("Device for training: cpu")
    device = torch.device(dev)

    # Initialise Models
    central_network = Net()
    global_network = Net()

    central_network.load_state_dict(global_network.state_dict())
    central_network.to(device)

    networks = [Net() for _ in range(num_clients)]
    for network in networks:
        network.load_state_dict(global_network.state_dict())
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
                                                     [int(cifar_train.data.shape[0] / 4) for _ in range(4)])

    train_loaders = [torch.utils.data.DataLoader(dataset=x,
                                                 batch_size=batch_size,
                                                 num_workers=4,
                                                 shuffle=True) for x in train_data_split]

    central_loader = torch.utils.data.DataLoader(dataset=cifar_train,
                                                 batch_size=batch_size,
                                                 num_workers=4,
                                                 shuffle=True)

    # Initialize loss as CrossEntropyLoss
    crosslosses = [nn.CrossEntropyLoss() for _ in networks]
    central_crossloss = nn.CrossEntropyLoss()

    # Initialize optimizer as SDG-optimizer
    optimizer = [torch.optim.SGD(network.parameters(), lr=0.03, momentum=0.9) for network in networks]
    central_optimizer = torch.optim.SGD(central_network.parameters(), lr=0.03, momentum=0.9)

    '''
    Global:
        for t=1 communication_round:
            call clients and wait
            avg. parameters
            share clients
            
    Client:
        for epoch=10
            for batches
                train
            end for
        end for
        
        send parameter
    '''

    federated_loss = [{'epoch_loss': [], 'batch_loss': []} for _ in range(num_clients)]
    central_loss = {'epoch_loss': [], 'batch_loss': []}

    # Federated training
    print("Start federated training")
    for t in range(communication_rounds):
        print('--------------------')
        print(f'Start communication round {t}:')
        for idx in range(num_clients):
            print(f'Client {idx} training:')
            loss_tuple = local_learner(device, epochs,
                                       train_loaders[idx],
                                       networks[idx],
                                       optimizer[idx],
                                       crosslosses[idx])
            # federated_loss[id].update(federated_loss[id].get('batch_loss') + (loss_tuple.get('batch_loss')))
            federated_loss[idx].get('batch_loss').extend(loss_tuple.get('batch_loss'))
            federated_loss[idx].get('epoch_loss').extend(loss_tuple.get('epoch_loss'))

        # Load parameters from individual networks and average in global network
        global_dict = global_network.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([networks[i].state_dict()[k].float() for i in range(len(networks))], 0).mean(0)
        global_network.load_state_dict(global_dict)
        # Load new parameters to individual networks
        for network in networks:
            network.load_state_dict(global_network.state_dict())

        print('--------------------')

    # Central training
    print("Start central training")
    loss_tuple = local_learner(device,
                               epochs * communication_rounds,
                               central_loader,
                               central_network,
                               central_optimizer,
                               central_crossloss)
    central_loss.get('batch_loss').extend(loss_tuple.get('batch_loss'))
    central_loss.get('epoch_loss').extend(loss_tuple.get('epoch_loss'))

    # Load test data to cpu (gpu memory would overflow)
    test_loader = torch.utils.data.DataLoader(dataset=cifar_test,
                                              batch_size=len(cifar_test),
                                              shuffle=True)
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
    plt.show()

    plt.title('Central Network: loss per batch')
    plt.xlabel('batch_id')
    plt.ylabel('batch_loss')
    plt.plot([i for i in range(0, len(central_loss.get('batch_loss')))], central_loss.get('batch_loss'),
             label='Central Network')
    plt.legend()
    plt.show()

    # Plotting epoch loss
    epoch_loss_length = 0
    for loss_tuple in federated_loss:
        epoch_loss = loss_tuple.get('epoch_loss')
        epoch_loss_length = len(epoch_loss)
        plt.plot([i for i in range(0, len(epoch_loss))], epoch_loss)

    for communication_round in range(communication_rounds-1):
        plt.axvline(x=epoch_loss_length/communication_rounds*(communication_round+1)-0.5, color='black')

    plt.plot([i for i in range(0, len(central_loss.get('epoch_loss')))], central_loss.get('epoch_loss'),
             label='Central Network')

    plt.title('Loss per epoch')
    plt.xlabel('epoch_id')
    plt.ylabel('epoch_loss')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([communication_line])
    plt.legend(handles=handles)
    plt.show()

    # Test global network
    for x_test, y_test in test_loader:
        global_network.to('cpu')
        prediction = global_network(x_test)
        correct_prediction = (torch.max(prediction.data, dim=1)[1] == y_test.data)
        accuracy = correct_prediction.float().mean().item()
        s = '\n Global Accuracy: {:2.2f}%'.format(accuracy * 100)
        print(s)

    # Test central network
    for x_test, y_test in test_loader:
        central_network.to('cpu')
        prediction = central_network(x_test)
        correct_prediction = (torch.max(prediction.data, dim=1)[1] == y_test.data)
        accuracy = correct_prediction.float().mean().item()
        s = '\n Central Accuracy: {:2.2f}%'.format(accuracy * 100)
        print(s)


def local_learner(device, epoch, loader, network, optimizer, crossloss):

    epoch_loss = []
    batch_loss = []
    # Start training
    for epoch in range(epoch):
        for i, (batch_X, batch_Y) in enumerate(loader):
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
                current_loss = loss.cpu().detach().numpy()
                batch_loss.append(current_loss)

        # print progress
        if epoch % 1 == 0:
            s = f'-> Epoch: {epoch + 1} completed. Current loss: {current_loss} '
            print(s)

        epoch_loss.append(batch_loss[-1])

    return {'epoch_loss': epoch_loss, 'batch_loss': batch_loss}


def main():
    train(num_clients=2, epochs=3, communication_rounds=4, batch_size=64)


if __name__ == '__main__':
    main()
