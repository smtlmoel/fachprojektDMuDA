import threading
from threading import Thread
import torch


class ClientThread(Thread):
    # noinspection PyPep8Naming
    def __init__(self, threadID, epoch, loader, network, optimizer, crossloss, return_queue):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.epoch = epoch
        self.loader = loader
        self.network = network
        self.optimizer = optimizer
        #self.lr_scheduler = lr_scheduler
        self.crossloss = crossloss
        self.return_queue = return_queue

    def local_learner(self, device, epoch, loader, network, optimizer, crossloss):
        epoch_loss = []
        batch_loss = []

        network.train()
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
                s = f'Client {self.threadID} -> Epoch: {epoch + 1} completed. Current loss: {current_loss} '
                print(s)

            epoch_loss.append(batch_loss[-1])
            # lr_scheduler.step()

        return {'idx': self.threadID, 'epoch_loss': epoch_loss, 'batch_loss': batch_loss}

    def run(self):
        # Move network to device GPU or CPU
        if torch.cuda.is_available():
            dev = "cuda:0"
            # print("Device for training: cuda")
        else:
            dev = "cpu"
            # print("Device for training: cpu")
        device = torch.device(dev)

        self.return_queue.put(self.local_learner(device,
                                                 self.epoch,
                                                 self.loader,
                                                 self.network,
                                                 self.optimizer,
                                                 #self.lr_scheduler,
                                                 self.crossloss))


