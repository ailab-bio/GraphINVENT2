"""
Defines specific MPNN implementations.
"""
# load general packages and functions
from collections import namedtuple
import math
import torch

# load GraphINVENT-specific functions
import gnn.aggregation_mpnn
import gnn.edge_mpnn
import gnn.summation_mpnn
import gnn.modules


class GGNN(gnn.summation_mpnn.SummationMPNN):
    """
    The "gated-graph neural network" model.
    """
    def __init__(self, constants : namedtuple) -> None:
        super().__init__(constants)

        self.constants  = constants

        self.msg_nns    = torch.nn.ModuleList()
        for _ in range(self.constants.n_edge_features):
            self.msg_nns.append(
                gnn.modules.MLP(
                    in_features=self.constants.hidden_node_features,
                    hidden_layer_sizes=[self.constants.enn_hidden_dim] * self.constants.enn_depth,
                    out_features=self.constants.message_size,
                    dropout_p=self.constants.enn_dropout_p,
                )
            )

        self.gru        = torch.nn.GRUCell(
            input_size=self.constants.message_size,
            hidden_size=self.constants.hidden_node_features,
            bias=True
        )

        self.gather     = gnn.modules.GraphGather(
            node_features=self.constants.n_node_features,
            hidden_node_features=self.constants.hidden_node_features,
            out_features=self.constants.gather_width,
            att_depth=self.constants.gather_att_depth,
            att_hidden_dim=self.constants.gather_att_hidden_dim,
            att_dropout_p=self.constants.gather_att_dropout_p,
            emb_depth=self.constants.gather_emb_depth,
            emb_hidden_dim=self.constants.gather_emb_hidden_dim,
            emb_dropout_p=self.constants.gather_emb_dropout_p,
            big_positive=self.constants.big_positive
        )

        self.APDReadout = gnn.modules.GlobalReadout(
            node_emb_size=self.constants.hidden_node_features,
            graph_emb_size=self.constants.gather_width,
            mlp1_hidden_dim=self.constants.mlp1_hidden_dim,
            mlp1_depth=self.constants.mlp1_depth,
            mlp1_dropout_p=self.constants.mlp1_dropout_p,
            mlp2_hidden_dim=self.constants.mlp2_hidden_dim,
            mlp2_depth=self.constants.mlp2_depth,
            mlp2_dropout_p=self.constants.mlp2_dropout_p,
            f_add_elems=self.constants.len_f_add_per_node,
            f_conn_elems=self.constants.len_f_conn_per_node,
            f_term_elems=1,
            max_n_nodes=self.constants.max_n_nodes,
            device=self.constants.device,
        )

    def message_terms(self, nodes : torch.Tensor, node_neighbours : torch.Tensor,
                      edges : torch.Tensor) -> torch.Tensor:
        edges_v               = edges.view(-1, self.constants.n_edge_features, 1)
        node_neighbours_v     = edges_v * node_neighbours.view(-1,
                                                               1,
                                                               self.constants.hidden_node_features)
        terms_masked_per_edge = [
            edges_v[:, i, :] * self.msg_nns[i](node_neighbours_v[:, i, :])
            for i in range(self.constants.n_edge_features)
        ]
        return sum(terms_masked_per_edge)

    def update(self, nodes : torch.Tensor, messages : torch.Tensor) -> torch.Tensor:
        return self.gru(messages, nodes)

    def readout(self, hidden_nodes : torch.Tensor, input_nodes : torch.Tensor,
                node_mask : torch.Tensor) -> torch.Tensor:
        graph_embeddings = self.gather(hidden_nodes, input_nodes, node_mask)
        output           = self.APDReadout(hidden_nodes, graph_embeddings)
        return output
