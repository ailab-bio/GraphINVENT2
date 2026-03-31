"""
Defines specific MPNN implementations built on `SummationMPNN`.

  GGNN  -- Gated Graph Neural Network (Li et al., 2016)
"""

from collections import namedtuple

import gnn.modules
import gnn.summation_mpnn
import torch


class GGNN(gnn.summation_mpnn.SummationMPNN):
    """
    Gated Graph Neural Network (GGNN) for autoregressive molecular graph generation.

    Architecture overview
    ---------------------
    1. **Message passing** (repeated `message_passes` times):
       - One MLP per bond type computes a candidate message from each neighbour's
         hidden state, gated by the bond-type indicator.
       - All incoming messages are summed at each node.
       - A shared GRU cell updates each node's hidden state from the summed message.

    2. **Graph readout**:
       - `AttentionReadout` (graph gather) computes a single graph-level embedding
         by soft-weighting each node's hidden state by a learned attention score.
       - `ActionProbReadout` maps the per-node hidden states *and* the graph embedding to
         the flat, unnormalised Action Probability Distribution (action probabilities) logits.

    The action probabilities logits are later normalised (softmax) and used to sample the next
    construction step: add a new node, connect two existing nodes, or terminate.

    Args:
        constants: Experiment constants namedtuple.  The fields consumed here are:
            n_edge_features, hidden_node_features, enn_hidden_dim, enn_depth,
            enn_dropout_p, message_size, gather_width, gather_att_depth,
            gather_att_hidden_dim, gather_att_dropout_p, gather_emb_depth,
            gather_emb_hidden_dim, gather_emb_dropout_p, big_positive,
            mlp1_hidden_dim, mlp1_depth, mlp1_dropout_p, mlp2_hidden_dim,
            mlp2_depth, mlp2_dropout_p, len_f_add_per_node, len_f_conn_per_node,
            max_n_nodes, device.
    """

    def __init__(self, constants: namedtuple) -> None:
        super().__init__(constants)

        self.constants = constants

        # One edge-network MLP per bond type: transforms a neighbour's hidden
        # state into a message vector, gated by the corresponding bond indicator.
        self.msg_nns = torch.nn.ModuleList(
            gnn.modules.MLP(
                in_features=self.constants.hidden_node_features,
                hidden_layer_sizes=[self.constants.enn_hidden_dim]
                * self.constants.enn_depth,
                out_features=self.constants.message_size,
                dropout_p=self.constants.enn_dropout_p,
            )
            for _ in range(self.constants.n_edge_features)
        )

        # GRU cell that updates each node's hidden state from aggregated messages.
        self.gru = torch.nn.GRUCell(
            input_size=self.constants.message_size,
            hidden_size=self.constants.hidden_node_features,
            bias=True,
        )

        # Attention-weighted graph pooling → single graph embedding vector.
        self.gather = gnn.modules.AttentionReadout(
            node_features=self.constants.n_node_features,
            hidden_node_features=self.constants.hidden_node_features,
            out_features=self.constants.gather_width,
            att_depth=self.constants.gather_att_depth,
            att_hidden_dim=self.constants.gather_att_hidden_dim,
            att_dropout_p=self.constants.gather_att_dropout_p,
            emb_depth=self.constants.gather_emb_depth,
            emb_hidden_dim=self.constants.gather_emb_hidden_dim,
            emb_dropout_p=self.constants.gather_emb_dropout_p,
            big_positive=self.constants.big_positive,
        )

        # Two-tier readout that predicts the full action probabilities from node + graph embeddings.
        self.ActionProbReadout = gnn.modules.ActionProbReadout(
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

    def message_terms(
        self, nodes: torch.Tensor, node_neighbours: torch.Tensor, edges: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes one message vector per directed edge.

        For each bond type i, the neighbour's hidden state is passed through
        `msg_nns[i]` and then masked by the bond-type indicator.  The results
        across all bond types are summed to give a single message per edge.

        Args:
            nodes:           Hidden states of source nodes (unused here; the GGNN
                             formulation conditions messages only on the *neighbour*).
                             Shape: (n_edges, hidden_node_features)
            node_neighbours: Hidden states of neighbouring nodes.
                             Shape: (n_edges, hidden_node_features)
            edges:           One-hot bond-type features for each edge.
                             Shape: (n_edges, n_edge_features)

        Returns:
            messages: One message vector per edge.
                      Shape: (n_edges, message_size)
        """
        edges_v = edges.view(-1, self.constants.n_edge_features, 1)
        node_neighbours_v = edges_v * node_neighbours.view(
            -1, 1, self.constants.hidden_node_features
        )
        terms_per_bond_type = [
            edges_v[:, i, :] * self.msg_nns[i](node_neighbours_v[:, i, :])
            for i in range(self.constants.n_edge_features)
        ]
        return sum(terms_per_bond_type)

    def update(self, nodes: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """
        Updates node hidden states using a GRU cell.

        Args:
            nodes:    Current hidden states.
                      Shape: (n_nodes_with_neighbours, hidden_node_features)
            messages: Summed incoming messages for each node.
                      Shape: (n_nodes_with_neighbours, message_size)

        Returns:
            Updated hidden states. Same shape as `nodes`.
        """
        return self.gru(messages, nodes)

    def readout(
        self,
        hidden_nodes: torch.Tensor,
        input_nodes: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Produces the action probabilities logits from the final node hidden states.

        Args:
            hidden_nodes: Final hidden states after all message-passing rounds.
                          Shape: (batch, max_n_nodes, hidden_node_features)
            input_nodes:  Original node feature vectors.
                          Shape: (batch, max_n_nodes, n_node_features)
            node_mask:    True for real nodes, False for padding.
                          Shape: (batch, max_n_nodes)

        Returns:
            Flat, unnormalised action probabilities logits.
            Shape: (batch, len_f_add + len_f_conn + 1)
        """
        graph_embeddings = self.gather(hidden_nodes, input_nodes, node_mask)
        return self.ActionProbReadout(hidden_nodes, graph_embeddings)
