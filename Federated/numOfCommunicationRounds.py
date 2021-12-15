import argparse

import federatedCNN
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--experiments', type=int, default=3, help='number of experiments')
    parser.add_argument('-epoch', type=int, default=10, help='epoch number for training')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-c', '--clients', type=int, default=4, help='number of clients for federated learning')
    parser.add_argument('-com_rounds', nargs="+", default=[2, 3, 4], help='communication round list(default: 2, 3, 4)')

    args = parser.parse_args()

    if args.experiments != len(args.com_rounds):
        print(args.experiments)
        print(len(args.com_rounds))
        print(args.com_rounds)
        print("Check number of experiments and and length of communication round list.")
        return

    for experiment in range(args.experiments):
        current_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        federatedCNN.train(num_clients=args.clients,
                           epochs=args.epoch,
                           communication_rounds=int(args.com_rounds[experiment]),
                           batch_size=args.batch_size,
                           experiment_name=f"numComRounds={args.com_rounds[experiment]}_epochs={args.epoch}_{current_time}")


if __name__ == '__main__':
    main()
