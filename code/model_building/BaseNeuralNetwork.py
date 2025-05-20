import torch.nn as nn

class BaseNeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, task_type='regression'):
        super().__init__()
        self.task_type = task_type

        # Define 10-layer fully connected neural network with ReLU activations between layers
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),  # Layer 1 - wider to capture more patterns
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout for regularization
            nn.BatchNorm1d(256),  # Add batch normalization for better training

            nn.Linear(256, 128),  # Layer 2
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),

            nn.Linear(128, output_dim)
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