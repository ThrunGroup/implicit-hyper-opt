import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.conv_input_model import ConvInputModel

"""
Conv network -> MLP architecture for classification of multimodal dataset
DESIGN:
     - three-layer MLP (relu activation + dropout on second layer)
     - uses log softmax loss & Adam optimizer 
"""
class CNN_MLP(nn.Module):
    def __init__(self, learning_rate):
        super(CNN_MLP, self).__init__()

        self.conv = ConvInputModel()
        self.fc1 = nn.Linear(5*5*24 + 11, 256)  # question (11 bits) concatenated to all
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 4)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, img, qst):
        x = self.conv(img) # x = (64 x 24 x 5 x 5)

        """fully connected layers"""
        x = x.view(x.size(0), -1)
        x = torch.cat((x, qst), 1)  # Concat question

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)
