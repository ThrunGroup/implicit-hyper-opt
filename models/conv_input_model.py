import torch.nn as nn
import torch.nn.functional as F

"""
Conv network for img encoding (mode 1)
DESIGN:
    - 4 conv layers with relu activation and batchnorm
"""
class ConvInputModel(nn.Module):
    def __init__(self):
        super(ConvInputModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 24, (3,3), stride=(2,2), padding=(1,1))
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, (3,3), stride=(2,2), padding=(1,1))
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, (3,3), stride=(2,2), padding=(1,1))
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, (3,3), stride=(2,2), padding=(1,1))
        self.batchNorm4 = nn.BatchNorm2d(24)

    def forward(self, img):
        #convolution
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x
