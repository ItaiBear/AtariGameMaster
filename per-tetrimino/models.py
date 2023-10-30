import torch
import torch.nn as nn



class TetrisNetwork(nn.Module):
    def __init__(self, input_dim=(20, 10)):
        super().__init__()
        print("input_dim: ", input_dim)
        self.frames = 1

        # CNN modeled off of Mnih et al.
        self.cnn = nn.Sequential(
            nn.Conv2d(self.frames, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU()
        )

        self.fc_layer_inputs = self.cnn_out_dim(input_dim)

        self.fully_connected = nn.Sequential(
            nn.Linear(self.fc_layer_inputs, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 1))

    def cnn_out_dim(self, input_dim):
        return self.cnn(torch.zeros(1, self.frames, *input_dim)
                        ).flatten().shape[0]

    def forward(self, x):
        cnn_out = self.cnn(x).reshape(-1, self.fc_layer_inputs)
        return self.fully_connected(cnn_out)