import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def main():
    # Plotting
    plt.style.use(['seaborn-dark-palette', 'ggplot'])
    plt.rcParams['figure.figsize'] = [10, 8]
    # First figure

    epochs = [1, 2, 3, 4, 5, 6, 10]
    data = [0.6751, 0.7344, 0.7405, 0.7449, 0.7431, 0.7466, 0.7464]
    plt.scatter(epochs, data)
    plt.plot(epochs, data, label="2 Clients")

    epochs = [1, 2, 3, 4, 5, 6, 10, 12, 15]
    data = [0.6094, 0.6750, 0.6788, 0.6897, 0.6815, 0.6903, 0.6827, 0.6860, 0.6899]
    plt.scatter(epochs, data)
    plt.plot(epochs, data, label="4 Clients")

    epochs = [1, 2, 3, 4, 5, 6, 10]
    data = [0.5452, 0.6044, 0.6156, 0.6198, 0.6199, 0.6192, 0.6185]
    plt.scatter(epochs, data)
    plt.plot(epochs, data, label="8 Clients")

    plt.ylim(0.5, 0.82)
    plt.xlim(0, 16)

    # Central : 0.7790
    plt.axhline(y=0.7790, color="black")
    central_line = mlines.Line2D([0], [0], color='black', label='Central')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([central_line])
    plt.legend(handles=handles)

    plt.title("60 Epochs")
    plt.xlabel("Number of communication rounds")
    plt.ylabel("Accuracy")

    plt.savefig("presentation/60epochs.png")
    plt.show()

    # Second Figure
    epochs = [1, 2, 3, 4, 5, 6]
    data = [0.5267, 0.6147, 0.6327, 0.6431, 0.6400, 0.6382]
    plt.scatter(epochs, data)
    plt.plot(epochs, data, label="Convolution layer 1")

    data = [0.1767, 0.2542, 0.2523, 0.2618, 0.2329, 0.2759]
    plt.scatter(epochs, data)
    plt.plot(epochs, data, label="Fully-connected layer 1")

    data = [0.1319, 0.2246, 0.2241, 0.2008, 0.2366, 0.2157]
    plt.scatter(epochs, data)
    plt.plot(epochs, data, label="ConvLayer1 and FCLayer 1")

    plt.ylim(0, 0.9)
    plt.xlim(0, 7)

    # Central : 0.7790
    plt.axhline(y=0.7790, color="black")
    central_line = mlines.Line2D([0], [0], color='black', label='Central')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([central_line])
    plt.legend(handles=handles)

    plt.title("Aggregation method: Skip layers")
    plt.xlabel("Number of communication rounds")
    plt.ylabel("Accuracy")

    plt.savefig("presentation/aggregation_layer.png")
    plt.show()

    # Third figure
    epochs = [1, 2, 3, 4, 5, 6]
    data = [0.1682, 0.3257, 0.4217, 0.4521, 0.5259, 0.5774]
    plt.scatter(epochs, data)
    plt.plot(epochs, data, label="mask 33%")

    data = [0.2794, 0.4967, 0.5795, 0.6016, 0.6206, 0.6232]
    plt.scatter(epochs, data)
    plt.plot(epochs, data, label="mask 50%")

    data = [0.3963, 0.5914, 0.6465, 0.6416, 0.6531, 0.6517]
    plt.scatter(epochs, data)
    plt.plot(epochs, data, label="mask 67%")

    plt.ylim(0, 0.9)
    plt.xlim(0, 7)

    # Central : 0.7790
    plt.axhline(y=0.7790, color="black")
    central_line = mlines.Line2D([0], [0], color='black', label='Central')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([central_line])
    plt.legend(handles=handles)

    plt.title("Aggregation method: mask parameters")
    plt.xlabel("Number of communication rounds")
    plt.ylabel("Accuracy")

    plt.savefig("presentation/aggregation_mask.png")
    plt.show()

    # Fourth Figure
    epochs = [1, 2, 3, 5, 10, 15, 30, 50, 75, 100]
    data = [0.5045, 0.7120, 0.7356, 0.7534, 0.7580, 0.7652, 0.7773, 0.7773, 0.7759, 0.7785]
    plt.scatter(epochs, data)
    plt.plot(epochs, data, label="4 Clients")

    plt.ylim(0.4, 0.82)

    # Central : 0.79.97
    plt.axhline(y=0.7997, color="black")
    central_line = mlines.Line2D([0], [0], color='black', label='Central')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([central_line])
    plt.legend(handles=handles)

    plt.title("300 Epochs")
    plt.xlabel("Number of communication rounds")
    plt.ylabel("Accuracy")

    plt.savefig("presentation/300epochs.png")
    plt.show()

if __name__ == '__main__':
    main()
