import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_mlp(dataset):
    """
    Train a simple MLP on the given dataset.
    Design your own model architecture, and choose a loss function and an optimizer.

    Args:
        dataset (torch.utils.data.Dataset): x is d-dimensional, y is 1-dimensional.

    Returns:
        model (MLP): trained model.
    """
    # YOUR CODE HERE
    model = MLP(dataset.x.shape[1], 100, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32)

    for epoch in range(100):
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

    return model
