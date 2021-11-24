from torch import nn as nn


class CIFARNet(nn.Module):
    def __init__(self):
        super(CIFARNet, self).__init__()
        '''
        ConvLayer 0
        ImageIn (32x32x3)
        -> Conv2d (32x32x96)
        -> ReLu
        -> MaxPool (16x16x96)
        '''
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2)
        )
        '''
        ConvLayer 1
        ImageIn (16x16x96)
        -> Conv2d (16x16x80)
        -> ReLu
        -> MaxPool (8x8x80)
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 80, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2)
        )
        '''
        ConvLayer 2
        ImageIn (8x8x80)
        -> Conv2d (8x8x96)
        -> ReLu
        -> Conv2d (8x8x64)
        -> ReLu
        '''
        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(3, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU()
        )
        '''
        ImageIn (8x8x64)
        -> Flatten
        -> Output (4096)
        '''
        self.flat = nn.Flatten()
        '''
        Input (4096)
        -> Relu
        -> Output (256)
        '''
        self.dense0 = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU()
        )
        '''
        Input (256)
        -> Relu
        -> Output (10)
        '''
        self.dense1 = nn.Sequential(
            nn.Linear(256, 10),
            nn.ReLU(),
            nn.Dropout()
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flat(x)
        x = self.dense0(x)
        out = self.dense1(x)
        return out
