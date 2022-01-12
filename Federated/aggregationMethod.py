import argparse
import math
import traceback
import torch
import federatedCNN
import centralCNN
from datetime import datetime
from CNNNet import Net


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--experiments', type=int, default=4, help='number of experiments')
    parser.add_argument('-epoch', type=int, default=60, help='epoch number for training')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-c', '--clients', type=int, default=4, help='number of clients for federated learning')
    parser.add_argument('-aggregation', type=str, default='normal',
                        help='method for aggregation parameters from client networks')
    parser.add_argument('-com_rounds', nargs="+", default=[1, 2, 3, 5],
                        help='communication round list(default: 1, 2, 3, 5)')

    args = parser.parse_args()

    if args.experiments != len(args.com_rounds):
        print(args.experiments)
        print(len(args.com_rounds))
        print(args.com_rounds)
        print("Check number of experiments and length of communication round list.")
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

    # Central
    print("-----Central CNN-----")
    current_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    print(f"Start time: {current_time}")
    print("{:<10} {:<10}".format('Epochs', args.epoch))
    print("{:<10} {:<10}".format('Batch_Size', args.batch_size))
    print("{:<10} {:<10}".format('Clients', args.clients))
    print("{:<10} {:<10}".format('Com-rounds', 0))
    centralCNN.train(epochs=args.epoch,
                     batch_size=args.batch_size,
                     experiment_name=f"centralNetwork_epoch={args.epoch}_{current_time}")

    # Federated
    for experiment in range(args.experiments):
        print(f"-----Federated experiment {experiment}-----")
        current_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        print(f"Start time: {current_time}")
        print("{:<10} {:<10}".format('Epochs', args.epoch))
        print("{:<10} {:<10}".format('Batch_Size', args.batch_size))
        print("{:<10} {:<10}".format('Clients', args.clients))
        print("{:<10} {:<10}".format('Com-rounds', int(args.com_rounds[experiment])))
        federatedCNN.train(num_clients=args.clients,
                           epochs=int(args.epoch / int(args.com_rounds[experiment])),
                           communication_rounds=int(args.com_rounds[experiment]),
                           batch_size=args.batch_size,
                           experiment_name=f"aggregation=normal_numComRounds={args.com_rounds[experiment]}_epochs={args.epoch}_{current_time}",
                           aggregation='normal')
        print("----------")
        federatedCNN.train(num_clients=args.clients,
                           epochs=int(args.epoch / int(args.com_rounds[experiment])),
                           communication_rounds=int(args.com_rounds[experiment]),
                           batch_size=args.batch_size,
                           experiment_name=f"aggregation=layer_numComRounds={args.com_rounds[experiment]}_epochs={args.epoch}_{current_time}",
                           aggregation='layers',
                           layer_list=["convLayer1.0.weight", "convLayer1.0.bias"])
        print("----------")
        global_dict = global_network.state_dict()
        mask = global_network.state_dict()
        for k in global_dict:
            rand_mat = torch.rand(size=global_dict[k].data.shape).to(device)
            bool_tensor = rand_mat <= torch.kthvalue(torch.flatten(rand_mat), round(0.5 * math.prod(rand_mat.shape)))[0]
            mask[k].data = torch.where(bool_tensor, torch.tensor(1).to(device), torch.tensor(0).to(device))

        federatedCNN.train(num_clients=args.clients,
                           epochs=int(args.epoch/int(args.com_rounds[experiment])),
                           communication_rounds=int(args.com_rounds[experiment]),
                           batch_size=args.batch_size,
                           experiment_name=f"aggregation=mask_numComRounds={args.com_rounds[experiment]}_epochs={args.epoch}_{current_time}",
                           aggregation='mask',
                           mask=mask)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc(file=open("logs/exception.txt", "a"))
        traceback.print_exc()
