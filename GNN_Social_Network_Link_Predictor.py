# Project: GNN_Social_Network_Link_Predictor.py
# Advanced Project 42: Graph Neural Network for Social Network Link Prediction

import torch
import torch_geometric as tg
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score

# --- (The core GNN model definition goes here) ---
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_node_features) # Output is original dimension

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

def run_gnn_link_prediction():
    print("Starting GNN Link Prediction Project...")
    
    # 1. Generate Synthetic or Load Real Graph Data (NetworkX)
    G = nx.erdos_renyi_graph(num_nodes=100, p=0.15, seed=42)
    
    # 2. Convert NetworkX graph to PyTorch Geometric Data format
    # x: Feature matrix (e.g., node features if available, otherwise identity matrix)
    x = torch.eye(G.number_of_nodes()) 
    
    # edge_index: Graph connectivity in COO format
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index)

    # 3. Model Initialization and Training Loop
    model = GCN(num_node_features=data.num_node_features, hidden_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # (Simplified training loop)
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        
        # Link prediction requires a decoder (e.g., simple dot product for similarity)
        # Using a placeholder loss for demonstration
        loss = torch.sum(z) * 0.01 # Placeholder: replace with BPR or binary cross-entropy on edges
        
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
    
    # 4. Evaluation (Link Prediction)
    model.eval()
    z = model(data.x, data.edge_index)
    
    # Simple evaluation: use dot product as link score
    # (Advanced: sample non-edges and calculate AUC)
    
    print("\nProject Finished: GNN trained for node embeddings.")
    # (Add a function to predict links between two specific nodes using the embedding z)

if __name__ == '__main__':
    # This project requires: pip install torch torch_geometric networkx scikit-learn
    try:
        run_gnn_link_prediction()
    except Exception as e:
        print(f"ERROR: Failed to run GNN project. Ensure dependencies are installed and check stack trace: {e}")
