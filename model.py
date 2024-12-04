#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        out = self.relu(out)
        return out

class ZeroModel(nn.Module):
    def __init__(self, board_size, action_shape, in_channels=1, backbone_layers=2):
        super(ZeroModel, self).__init__()
        # TODO: input a condition vector that represents the current rule of the game. 
        # so that the model can learn to play different games

        self.action_shape = action_shape
        self.board_size = board_size
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1)
        self.backbone = self._make_leyer(128, backbone_layers)

        self.policy_conv = nn.Conv2d(128, 3, kernel_size=1)
        self.policy_head = nn.Sequential(
            nn.Linear(3 * board_size * board_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_shape[0] * action_shape[1])
        )
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_head = nn.Sequential(
            nn.Linear(board_size * board_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.backbone(x)
        policy = self.policy_head(torch.reshape(self.policy_conv(x), (-1, 3 * self.board_size * self.board_size)))
        value = self.value_head(torch.reshape(self.value_conv(x), (-1, self.board_size * self.board_size)))
        policy = policy.view(-1, self.action_shape[0], self.action_shape[1])
        return policy, value
    
    def _make_leyer(self, channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(channels, channels))
        return nn.Sequential(*layers)

#%%
if __name__ == '__main__':
    data = torch.randn(5, 1, 9, 9)
    model = ZeroModel(9, (1, 9), 1)
    print(model)
    output = model(data)
    print(output[0].shape, output[1].shape)
