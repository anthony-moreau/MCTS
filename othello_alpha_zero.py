import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(in_features)

    def forward(self, x: torch.Tensor):
        x1 = F.relu(self.batch_norm1(self.conv1(x)))
        x2 = self.batch_norm2(self.conv2(x1))
        return x + x2

# optim from internet > no bias + 2 ReLu instead of 1
# class ResBlock(nn.Module):
#     def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
#         super(ResBlock, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#
#     def forward(self, x):
#         residual = x
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += residual
#         out = F.relu(out)
#         return out


class OthelloZero(nn.Module):

    def __init__(self, n_input, n_filter, n_residual_block):
        super(OthelloZero, self).__init__()
        # convolutionnal block
        self.conv = nn.Conv2d(n_input, n_filter, kernel_size=(2,2))
        self.conv_batch_norm = nn.BatchNorm2d(n_filter)
        #residual blocks
        self.residual_blocks = [ResidualBlock(in_features=n_filter) for i in range(n_residual_block)]
        #policy head
        self.policy_conv = nn.Conv2d(n_filter, 2, kernel_size=(1, 1))
        self.policy_batch_norm = nn.BatchNorm2d(2)
        self.policy_fully_connected = nn.Linear(50, 33)
        #value head
        self.value_conv = nn.Conv2d(n_filter, 1, kernel_size=(1,1))
        self.value_batch_norm = nn.BatchNorm2d(1)
        self.fully1 = nn.Linear(25, 256)
        self.fully2 = nn.Linear(256,1)

    def forward(self, x: torch.Tensor):
        x1 = F.relu(self.conv_batch_norm(self.conv(x)))
        for block in self.residual_blocks:
            x1 = block(x1)
        policy = F.relu(self.policy_batch_norm(self.policy_conv(x1))).flatten()
        policy = nn.Softmax(self.policy_fully_connected(policy))
        value = F.relu(self.value_batch_norm(self.value_conv(x1))).flatten()
        value = torch.tanh(self.fully2(F.relu(self.fully1(value))))
        return policy, value


# Othello 6*6 grid
grid = [6 * [0] for i in range(6)]
grid[2][2] = 1
grid[2][3] = 2
grid[3][2] = 2
grid[3][3] = 1

tensor = torch.Tensor([[grid for i in range(3)]])

test = OthelloZero(3, 256, 2)
print(test(tensor))
print("hello")