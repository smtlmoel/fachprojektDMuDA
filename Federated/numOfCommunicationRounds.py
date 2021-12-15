import argparse
import traceback
import torch
import federatedCNN
import centralCNN
from datetime import datetime

from CNNNet import Net


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--experiments', type=int, default=9, help='number of experiments')
    parser.add_argument('-epoch', type=int, default=60, help='epoch number for training')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-c', '--clients', type=int, default=4, help='number of clients for federated learning')
    parser.add_argument('-com_rounds', nargs="+", default=[1, 2, 3, 4, 5, 6, 10, 12, 15], help='communication round list(default: 2, 3, 4)')

    args = parser.parse_args()

    if args.experiments != len(args.com_rounds):
        print(args.experiments)
        print(len(args.com_rounds))
        print(args.com_rounds)
        print("Check number of experiments and and length of communication round list.")
        return

    if torch.cuda.is_available():
        dev = "cuda:0"
        print("Device for training: cuda")
    else:
        dev = "cpu"
        print("Device for training: cpu")
    device = torch.device(dev)

    # Initiate global net
    global_network = Net()
    global_network.to(device)
    torch.save(global_network, f"models/initial_weights.pth")

    # Print parameter memory size
    mem_params = sum([param.nelement()*param.element_size() for param in global_network.parameters()])
    print(f'Memory Parameters: {mem_params/1024} kB')

    # Central
    current_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    centralCNN.train(epochs=args.epoch,
                     batch_size=args.batch_size,
                     experiment_name=f"centralNetwork_epoch={args.epoch}_{current_time}")

    # Federated
    for experiment in range(args.experiments):
        current_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        federatedCNN.train(num_clients=args.clients,
                           epochs=int(args.epoch/int(args.com_rounds[experiment])),
                           communication_rounds=int(args.com_rounds[experiment]),
                           batch_size=args.batch_size,
                           experiment_name=f"numComRounds={args.com_rounds[experiment]}_epochs={args.epoch}_{current_time}")


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc(file=open("logs/exception.txt", "a"))
        traceback.print_exc()
