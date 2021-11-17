from torch import nn as nn


class Net(nn.Module):
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
