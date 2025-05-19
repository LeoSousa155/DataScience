import torch.nn as nn

class BaseNeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, task_type='regression'):
        super().__init__()
        self.task_type = task_type

        # Define 10-layer fully connected neural network with ReLU activations between layers
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),  # Layer 1
            nn.ReLU(),
            nn.Linear(128, 128),        # Layer 2
            nn.ReLU(),
            nn.Linear(128, 64),         # Layer 3
            nn.ReLU(),
            nn.Linear(64, 64),          # Layer 4
            nn.ReLU(),
            nn.Linear(64, 64),          # Layer 5
            nn.ReLU(),
            nn.Linear(64, 32),          # Layer 6
            nn.ReLU(),
            nn.Linear(32, 32),          # Layer 7
            nn.ReLU(),
            nn.Linear(32, 16),          # Layer 8
            nn.ReLU(),
            nn.Linear(16, 16),          # Layer 9
            nn.ReLU(),
            nn.Linear(16, output_dim)   # Layer 10 (output layer)
        )

        # For classification, use softmax on output
        if task_type == 'classification':
            self.activation = nn.Softmax(dim=1)
        else:
            self.activation = None

    def forward(self, x):
        out = self.model(x)
        if self.activation:
            out = self.activation(out)
        return out