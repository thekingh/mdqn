import torch.nn as nn
import torch.nn.functional as F

'''
class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Linear(7 * 7 * 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.layers(x)
'''

class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1)
        self.fc4 = nn.Linear(8 * 8 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)
