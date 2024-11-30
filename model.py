#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

class ZeroModel(nn.Module):
    def __init__(self, board_size, in_channels=1):
        super(ZeroModel, self).__init__()
        # TODO: input a condition vector that represents the current rule of the game. so that the model can learn to play different games
        # TODO: delete the max pooling layers
        self.board_size = board_size
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.policy_head = nn.Sequential(
            nn.Linear(128 * (board_size // 6), 64),
            nn.ReLU(),
            nn.Linear(64, board_size * board_size)
        )
        self.value_head = nn.Sequential(
            nn.Linear(128 * (board_size // 6), 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        value = self.value_head(x)
        policy = self.policy_head(x)
        policy = F.softmax(policy, dim=1)
        policy = policy.view(-1, self.board_size, self.board_size)
        return policy, value
    



#%%
if __name__ == '__main__':
    data = torch.randn(5, 1, 9, 9)
    model = ZeroModel(9)
    print(model)
    output = model(data)
    print(output[0].shape, output[1].shape)
