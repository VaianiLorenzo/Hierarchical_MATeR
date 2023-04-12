import torch
from torch import nn, Tensor
from transformers import AutoModelForSequenceClassification
from mlp import MLP
from hierarchical_transformer import h_transformer

class Hierarchical_MATeR(nn.Module):
    
    def __init__(self, d_model: int = 1536, nhead: int = 4, d_hid: int = 768,
                 nlayers: int = 1, dropout: float = 0.25, device=None):
        super().__init__()
        self.h_transformer = h_transformer(d_model, nhead, d_hid, nlayers, dropout)
        self.d_model = d_model
        self.mlp = MLP(input_dim=d_model, output_dim = 1)
        self.device = device
        self.mlp = self.mlp.to(self.device)
        
    def forward(self, embeddings, attention_masks):
        
        # compute hierarchical encoding
        hierarchical_encoding = self.h_transformer(embeddings, attention_masks)

        # flatten the hierarchical encoding
        hierarchical_encoding = hierarchical_encoding.view(-1, self.d_model)

        # flatten the attention masks
        attention_masks = attention_masks.view(-1)
        
        # get non-padding encodings
        hierarchical_encoding = hierarchical_encoding[attention_masks == 1]

        # pass to MLP
        output = self.mlp(hierarchical_encoding)

        return output