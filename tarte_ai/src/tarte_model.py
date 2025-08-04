"""
TARTE neural network model used for pretraining and downstream tasks.
"""

import torch
import torch.nn as nn
from typing import Union


## TARTE - Base Model
class TARTE_Base(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_transformer: int,
        dim_feedforward: int,
        num_heads: int,
        num_layers_transformer: int,
        dropout: float,
    ):
        super(TARTE_Base, self).__init__()

        # Initial linear layers for cell features
        self.initial_x = nn.Sequential(
            nn.Linear(dim_input, dim_transformer),
            nn.ReLU(),
            nn.LayerNorm(dim_transformer),
        )

        # Initial linear layers for column features
        self.initial_e = nn.Sequential(
            nn.Linear(dim_input, dim_transformer),
            nn.ReLU(),
            nn.LayerNorm(dim_transformer),
        )

        # Initial linear layers for cell features
        self.readout = nn.Sequential(
            nn.Linear(1, dim_transformer),
            nn.ReLU(),
            nn.LayerNorm(dim_transformer),
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_transformer,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            bias=True,
            batch_first=True,
            norm_first=True,
            # activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers_transformer,
            enable_nested_tensor=False,
        )

    def forward(self, x, edge_attr, mask):

        # Initial linear layers for cell and column features
        x = self.initial_x(x)
        edge_attr = self.initial_e(edge_attr)

        # Combine the cell and column features with addition
        z = x + edge_attr

        # Initialization of the readout
        readout_node = self.readout(torch.ones(z.size(0), 1, device=x.device))
        z[:, 0, :] = readout_node

        # Feed forward on tranformers
        z = self.transformer_encoder(z, src_key_padding_mask=mask)

        return z


## TARTE - Pretrain Model
class TARTE_Pretrain_NN(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_transformer: int,
        dim_feedforward: int,
        dim_projection: Union[int, list],
        num_heads: int,
        num_layers_transformer: int,
        dropout: float,
    ):
        super(TARTE_Pretrain_NN, self).__init__()

        if isinstance(dim_projection, int):
            self.dim_projection = [dim_projection]
        else:
            self.dim_projection = dim_projection

        # Base layer (transformer)
        self.tarte_base = TARTE_Base(
            dim_input=dim_input,
            dim_transformer=dim_transformer,
            dim_feedforward=dim_feedforward,
            num_heads=num_heads,
            num_layers_transformer=num_layers_transformer,
            dropout=dropout,
        )

        self.layer_norm = nn.LayerNorm(dim_transformer)

        self.tarte_linear = torch.nn.ModuleDict(
            {
                f"proj_{d_proj}": nn.Sequential(
                    nn.Linear(dim_transformer, d_proj),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                    nn.LayerNorm(d_proj),
                    nn.Linear(d_proj, dim_feedforward),
                    nn.ReLU(),
                    nn.LayerNorm(dim_feedforward),
                    nn.Linear(dim_feedforward, d_proj),
                )
                for d_proj in self.dim_projection
            }
        )

    def forward(self, x, edge_attr, mask):

        # Feed-forward on TARTE Base layer
        x = self.tarte_base(x, edge_attr, mask)
        x = self.layer_norm(x)

        # Linear projections
        output = []
        for lin_proj in self.tarte_linear.values():
            out = lin_proj(x)
            output.append(out[:, 0, :])

        return output


## TARTE - Downstream Model
class TARTE_Downstream_NN(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_transformer: int,
        dim_feedforward: int,
        dim_output: int,
        num_heads: int,
        num_layers_transformer: int,
        dropout: float,
    ):
        super(TARTE_Downstream_NN, self).__init__()

        self.tarte_base = TARTE_Base(
            dim_input=dim_input,
            dim_transformer=dim_transformer,
            dim_feedforward=dim_feedforward,
            num_heads=num_heads,
            num_layers_transformer=num_layers_transformer,
            dropout=dropout,
        )

        self.layer_norm = nn.LayerNorm(dim_transformer)

        self.classifier = nn.Sequential(
            nn.Linear(dim_transformer, int(dim_transformer / 2)),
            nn.ReLU(),
            nn.LayerNorm(int(dim_transformer / 2)),
            nn.Linear(int(dim_transformer / 2), int(dim_transformer / 4)),
            nn.ReLU(),
            nn.LayerNorm(int(dim_transformer / 4)),
            nn.Linear(int(dim_transformer / 4), dim_output),
        )

    def forward(self, x, edge_attr, mask):

        x = self.tarte_base(x, edge_attr, mask)
        x = self.layer_norm(x)

        x = x[:, 0, :]
        x = self.classifier(x)

        return x
