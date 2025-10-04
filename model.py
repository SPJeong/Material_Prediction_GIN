##### model.py (GIN)

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GINConv


class GIN_Model(nn.Module):
    def __init__(self, in_channels=57, out_channels=1024, hidden_dim=256, descriptor_ecfp2_size=1217, dropout=0.2):
        super().__init__()
        # GIN
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        nn1 = nn.Sequential(nn.Linear(in_channels, in_channels * 2), nn.ReLU(),
                            nn.Linear(in_channels * 2, in_channels * 2))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(in_channels * 2)

        nn2 = nn.Sequential(nn.Linear(in_channels * 2, in_channels * 4), nn.ReLU(),
                            nn.Linear(in_channels * 4, in_channels * 4))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(in_channels * 4)

        nn3 = nn.Sequential(nn.Linear(in_channels * 4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)

        nn4 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(hidden_dim)

        nn5 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(), nn.Linear(hidden_dim * 2, hidden_dim * 2))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(hidden_dim * 2)

        # gin last layer
        self.gin_fc = nn.Linear(hidden_dim * 2, out_channels)

        # final FC
        self.final_fc1 = nn.Linear(out_channels + descriptor_ecfp2_size, 1024)  # (1024+1217) -> 1024
        self.final_fc2 = nn.Linear(1024, 2048)
        self.final_fc3 = nn.Linear(2048, 1024)
        self.final_fc4 = nn.Linear(1024, 256)
        self.final_fc5 = nn.Linear(256, 1)

    def forward(self, data_graph, data_descriptor_ECFP):
        x, edge_index, batch = data_graph.x, data_graph.edge_index, data_graph.batch

        # GIN forward
        x = self.bn1(self.conv1(x, edge_index))
        x = self.bn2(self.conv2(x, edge_index))
        x = self.bn3(self.conv3(x, edge_index))
        x = self.bn4(self.conv4(x, edge_index))
        x = self.bn5(self.conv5(x, edge_index))

        # Global pooling (can choose one btween two)
        x = torch_geometric.nn.global_add_pool(x, batch)  # better for molecule's property prediction
        # x = torch_geometric.nn.global_mean_pool(x, batch)

        # GCN fc
        x = self.gin_fc(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # concat GIN + descriptor_ECFP
        combined_features = torch.cat([x, data_descriptor_ECFP], dim=1)

        # final FC
        out = self.relu(self.final_fc1(combined_features))
        out = self.dropout(out)
        out = self.relu(self.final_fc2(out))
        out = self.dropout(out)
        out = self.relu(self.final_fc3(out))
        out = self.dropout(out)
        out = self.relu(self.final_fc4(out))
        out = self.final_fc5(out)

        return out


def count_parameters(model):
    """
    Calculates the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Calculate and print the total parameters
if __name__ == '__main__':
    my_deep_model = GIN_Model()
    total_params = count_parameters(my_deep_model)
    print(f"Total trainable parameters: {total_params:,}")