import traceback
from CNNNet import Net


def main():
    global_network = Net()

    output_file = open(f"networkDataUsage.txt", "w")

    mem_params = 0
    # Print parameter memory site per layer
    for name, param in global_network.named_parameters():
        layer_mem = param.nelement()*param.element_size()
        s = f'Memory of layer {name}: {layer_mem/1024} KiB'
        output_file.write(s+"\n")
        print(s)
        mem_params += layer_mem
    s = f'Total Memory: {mem_params/1024} KiB'
    output_file.write(s+"\n")
    print(s)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc(file=open("logs/exception.txt", "a"))
        traceback.print_exc()
