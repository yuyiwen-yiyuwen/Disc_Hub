import torch
import torch.nn  as nn

class ResNetModel(nn.Module):
    def __init__(self, input_dim):
        super(ResNetModel, self).__init__()
        self.dense1 = nn.Linear(input_dim, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.7)

        self.dense2 = nn.Linear(256, 256)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.7)

        self.dense3 = nn.Linear(256, 128)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.55)

        self.dense4 = nn.Linear(128, 64)
        self.batch_norm4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.45)

        self.dense5 = nn.Linear(64, 32)
        self.batch_norm5 = nn.BatchNorm1d(32)

        self.output = nn.Linear(32, 1)

    def forward(self, x, return_features=False):
        # Forward pass with residual connection
        x = torch.relu(self.batch_norm1(self.dense1(x)))
        x = self.dropout1(x)

        residual = torch.relu(self.batch_norm2(self.dense2(x)))
        residual = self.dropout2(residual)

        x = torch.add(x, residual)
        x = torch.relu(self.batch_norm3(self.dense3(x)))
        x = self.dropout3(x)

        x = torch.relu(self.batch_norm4(self.dense4(x)))
        x = self.dropout4(x)

        x = torch.relu(self.batch_norm5(self.dense5(x)))
        features = x
        out = torch.sigmoid(self.output(x))

        return (out, features) if return_features else out