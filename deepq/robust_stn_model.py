import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustSTN(nn.Module):

    def __init__(self, in_channels=4, num_actions=18):
        super(RobustSTN, self).__init__()
        #Previous Actions
        # 4*8. 4 states and 12 actions
        self.action_history = nn.Sequential(
                                nn.Linear(in_channels*num_actions, 48),
                                nn.LeakyReLU(negative_slope=0.2)
                                )

        #Screenshot History (input 4 images)
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
                                )

        #Localization Linear
        self.fc_loc1 = nn.Linear(192*3, 256, bias=False)
        self.bn_loc1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout2d(0.1)
        self.fc_loc2 = nn.Linear(256, 6, bias=False)

        # Subbranch with no Spatial Transformer
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(128)

        # Subbranch with Spatial Transformer
        self.conv4 = nn.Conv2d(1, 64, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(32)

        # Merge the 4 branches
        self.fc1 = nn.Linear(48 + 20736 + 1152 + 48672, 512)
        self.fc2 = nn.Linear(512, 12)

    def forward(self, x, y, z):
        # Action History
        x = x.view(-1, 48)
        x = self.action_history(x)
        
        # Screenshot History
        y = self.screenshot_history(y)

        # Processing for Last Screenshot
        z = z.unsqueeze(0)
        z = z.view(-1, 1, 84, 84)
        z0 = self.last_screenshot(z)

        # Last Screenshot without STN
        z1 = z0
        z1 = F.leaky_relu(self.bn1(self.conv1(z1)), 0.2)
        z1 = F.leaky_relu(self.bn2(self.conv2(z1)), 0.2)
        z1 = F.leaky_relu(self.bn3(self.conv3(z1)), 0.2)

        # Last Screenshot with STN
        z2 = self.localization(z0)
        z2 = z2.view(-1, 64*3*3)
        theta = self.fc_loc1(z2)
        theta = F.leaky_relu(theta, 0.2)
        theta = F.leaky_relu((self.fc_loc2(theta)), 0.2)
        theta = self.dropout1(theta)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, z.shape)
        z2 = F.grid_sample(z, grid)
        z2 = F.leaky_relu((self.bn4(self.conv4(z2))), 0.2)
        z2 = F.leaky_relu((self.bn5(self.conv5(z2))), 0.2)

        # Reshape for merging
        y = y.view(-1, 64*18*18)
        z1 = z1.view(-1, 128*3*3)
        z2 = z2.view(-1, 32*39*39)

        # Merge branches
        xyz = torch.cat([x, y, z1, z2], 1) 
        xyz = F.leaky_relu(self.fc1(xyz))
        xyz = self.fc2(xyz)
        return xyz