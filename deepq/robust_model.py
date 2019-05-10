import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustDQN(nn.Module):

    def __init__(self, in_channels=4, num_actions=18):
        super(RobustDQN, self).__init__()
        #Previous Actions
        # 4*8. 4 states and 12 actions
        self.action_history = nn.Sequential(
                                nn.Linear(in_channels*num_actions, 48),
                                #nn.BatchNorm1d(48),
                                nn.LeakyReLU(negative_slope=0.2)
                                )

        #Screenshot History (input 4 images). Replay Buffer

        self.screenshot_history = nn.Sequential(
                                    nn.Conv2d(in_channels, 32, kernel_size=3),
                                    nn.BatchNorm2d(32),
                                    nn.LeakyReLU(negative_slope=0.2),
                                    nn.Conv2d(32, 64, kernel_size=5, stride=2),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(negative_slope=0.2),
                                    nn.Conv2d(64, 64, kernel_size=5, stride=2),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(negative_slope=0.2),
                                    )
        #Last Screen Shot
        self.last_screenshot = nn.Sequential(
                                nn.Conv2d(1, 256, kernel_size=5),
                                nn.BatchNorm2d(256),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.MaxPool2d(kernel_size=2),
                                nn.Conv2d(256, 64, kernel_size=3),
                                nn.BatchNorm2d(64),
                                nn.LeakyReLU(negative_slope=0.2),
                                )

        #Localization Network for Spatial Transformer
        self.localization = nn.Sequential(
                                nn.Conv2d(64, 32, kernel_size=5, stride=2),
                                nn.BatchNorm2d(32),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Dropout2d(0.1),
                                nn.Conv2d(32, 64, kernel_size=3, stride=2),
                                nn.BatchNorm2d(64),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv2d(64, 64, kernel_size=3, stride=2),
                                nn.BatchNorm2d(64),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Dropout2d(0.2),
                                nn.Linear(2, 256),
                                nn.BatchNorm2d(256),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Dropout2d(0.2),
                                nn.Linear(256, 3),
                                nn.LeakyReLU(negative_slope=0.2)
                                )

        # Left branch no Spatial Transformer
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(128)

        # Right branch Spatial Transformer
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=2)
        self.bn5 = nn.BatchNorm2d(32)

        # Merge the 4 branches
        self.fc1 = nn.Linear(20784 + 1152, 512)
        self.bn6 = nn.BatchNorm2d(512)
        self.fc2 = nn.Linear(512, 12)


    def forward(self, x, y, z):
        x = x.view(-1, 48)
        x = self.action_history(x)
        
        y = self.screenshot_history(y)

        z = z.unsqueeze(0)
        z = z.view(-1, 1, 84, 84)
        z = self.last_screenshot(z)

        z1 = z
        z1 = F.leaky_relu(self.bn1(self.conv1(z1)), 0.2)
        z1 = F.leaky_relu(self.bn2(self.conv2(z1)), 0.2)
        z1 = F.leaky_relu(self.bn3(self.conv3(z1)), 0.2)

        '''
        z2 = self.localization(z)
        z2 = F.leaky_relu((self.bn4(self.conv4(z2))), 0.2)
        z2 = F.leaky_relu((self.bn5(self.conv5(z2))), 0.2)
        '''

        y = y.view(-1, 20736)
        z1 = z1.view(-1, 128*3*3)
        xyz = torch.cat([x, y, z1], 1) 
        xyz = self.fc1(xyz)
        xyz = self.fc2(xyz)
        return xyz