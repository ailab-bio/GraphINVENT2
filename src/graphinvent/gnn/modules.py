"""
Reusable neural network modules used by the GGNN model.

  MLP              -- multi-layer perceptron with SELU activations
  AttentionReadout -- attention-weighted graph-level pooling
  ActionProbReadout       -- predicts the full Action Probability Distribution (action probabilities)
"""

import torch


class AttentionReadout(torch.nn.Module):
    """
    Attention-weighted graph-level pooling, also known as "graph gather".

    Produces a single fixed-size embedding for the entire graph by computing
    a soft attention weight for each node and returning the weighted sum of
    node embeddings.  The attention scores are computed from a concatenation of
    each node's current hidden state and its original input features; this lets
    the network attend differently depending on both what has been learned and
    what the node originally looked like.

    Args:
        node_features:         Dimension of the original (input) node feature vectors.
        hidden_node_features:  Dimension of the hidden node state vectors.
        out_features:          Dimension of the output graph embedding.
        att_depth:             Number of layers in the attention MLP.
        att_hidden_dim:        Width of the attention MLP.
        att_dropout_p:         Dropout probability in the attention MLP.
        emb_depth:             Number of layers in the embedding MLP.
        emb_hidden_dim:        Width of the embedding MLP.
        emb_dropout_p:         Dropout probability in the embedding MLP.
        big_positive:          Large positive constant used to mask padding nodes
                               before the softmax (prevents padded positions from
                               receiving attention weight).
    """

    def __init__(
        self,
        node_features: int,
        hidden_node_features: int,
        out_features: int,
        att_depth: int,
        att_hidden_dim: int,
        att_dropout_p: float,
        emb_depth: int,
        emb_hidden_dim: int,
        emb_dropout_p: float,
        big_positive: float,
    ) -> None:
        super().__init__()

        self.big_positive = big_positive

        self.att_nn = MLP(
            in_features=node_features + hidden_node_features,
            hidden_layer_sizes=[att_hidden_dim] * att_depth,
            out_features=out_features,
            dropout_p=att_dropout_p,
        )
        self.emb_nn = MLP(
            in_features=hidden_node_features,
            hidden_layer_sizes=[emb_hidden_dim] * emb_depth,
            out_features=out_features,
            dropout_p=emb_dropout_p,
        )

    def forward(
        self,
        hidden_nodes: torch.Tensor,
        input_nodes: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_nodes: Final hidden states from message passing.
                          Shape: (batch, max_n_nodes, hidden_node_features)
            input_nodes:  Original node feature vectors.
                          Shape: (batch, max_n_nodes, n_node_features)
            node_mask:    True for real nodes, False for padding.
                          Shape: (batch, max_n_nodes)

        Returns:
            Graph-level embedding. Shape: (batch, out_features)
        """
        cat = torch.cat((hidden_nodes, input_nodes), dim=2)
        energy_mask = (node_mask == 0).float() * self.big_positive
        energies = self.att_nn(cat) - energy_mask.unsqueeze(-1)
        attention = torch.nn.functional.softmax(energies, dim=1)
        embedding = self.emb_nn(hidden_nodes)
        return torch.sum(attention * embedding, dim=1)


class MLP(torch.nn.Module):
    """
    Multi-layer perceptron with SELU activations and optional AlphaDropout.

    Each hidden layer is a `Linear → SELU → AlphaDropout` block.  Xavier
    uniform initialisation is applied to all weight matrices.  Bias terms are
    always included so that the network can learn from graphs that start as
    all-zeros (e.g. an empty molecule at the first generation step).

    Args:
        in_features:         Number of input features.
        hidden_layer_sizes:  List of widths for each hidden layer.
                             Pass an empty list for a single linear layer.
        out_features:        Number of output features.
        dropout_p:           AlphaDropout probability (0.0 = no dropout).
    """

    def __init__(
        self,
        in_features: int,
        hidden_layer_sizes: list,
        out_features: int,
        dropout_p: float,
    ) -> None:
        super().__init__()

        sizes = [in_features, *hidden_layer_sizes, out_features]
        layers = [
            self._linear_block(in_f, out_f, dropout_p)
            for in_f, out_f in zip(sizes, sizes[1:])
        ]
        # Flatten the list of Sequentials into a single Sequential
        self.seq = torch.nn.Sequential(
            *[module for sq in layers for module in sq.children()]
        )

    def _linear_block(
        self, in_f: int, out_f: int, dropout_p: float
    ) -> torch.nn.Sequential:
        linear = torch.nn.Linear(in_f, out_f, bias=True)
        torch.nn.init.xavier_uniform_(linear.weight)
        return torch.nn.Sequential(
            linear, torch.nn.SELU(), torch.nn.AlphaDropout(dropout_p)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ActionProbReadout(torch.nn.Module):
    """
    Predicts the Action Probability Distribution (action probabilities) for a batch of molecular graphs.

    The action probabilities encodes, for each graph, the probability of every possible next
    construction step:

      f_add[v, atom_type, charge, n_imp_H, bond_type]
          Probability of adding a new atom of a given type and bonding it to
          existing node v.

      f_conn[v, bond_type]
          Probability of adding a bond between two already-existing nodes,
          connecting to node v.

      f_term  (scalar)
          Probability of terminating the graph (declaring it complete).

    The readout uses a two-tier MLP architecture:
      Tier 1 (per-node):  fAddNet1 and fConnNet1 map each node's hidden state
                          to preliminary per-node action probabilities components.
      Tier 2 (per-graph): fAddNet2, fConnNet2, fTermNet2 refine the flattened
                          tier-1 outputs by also conditioning on the global
                          graph embedding from `AttentionReadout`.

    No activation is applied to the final output; callers are responsible for
    applying softmax (during training/loss) or sampling (during generation).

    Args:
        f_add_elems:   Number of elements in f_add per node
                       (= atom_types × charges × imp_H × bond_types).
        f_conn_elems:  Number of elements in f_conn per node (= bond_types).
        f_term_elems:  Always 1.
        mlp1_depth:    Depth of tier-1 MLPs.
        mlp1_dropout_p:   Dropout for tier-1 MLPs.
        mlp1_hidden_dim:  Width of tier-1 MLPs.
        mlp2_depth:    Depth of tier-2 MLPs.
        mlp2_dropout_p:   Dropout for tier-2 MLPs.
        mlp2_hidden_dim:  Width of tier-2 MLPs.
        graph_emb_size:   Dimension of the graph-level embedding.
        max_n_nodes:      Maximum number of nodes in any graph.
        node_emb_size:    Dimension of each node's hidden state.
        device:           'cuda' or 'cpu'.
    """

    def __init__(
        self,
        f_add_elems: int,
        f_conn_elems: int,
        f_term_elems: int,
        mlp1_depth: int,
        mlp1_dropout_p: float,
        mlp1_hidden_dim: int,
        mlp2_depth: int,
        mlp2_dropout_p: float,
        mlp2_hidden_dim: int,
        graph_emb_size: int,
        max_n_nodes: int,
        node_emb_size: int,
        device: str,
    ) -> None:
        super().__init__()

        self.device = device

        # Tier 1 — per-node preliminary distributions
        self.fAddNet1 = MLP(
            in_features=node_emb_size,
            hidden_layer_sizes=[mlp1_hidden_dim] * mlp1_depth,
            out_features=f_add_elems,
            dropout_p=mlp1_dropout_p,
        )
        self.fConnNet1 = MLP(
            in_features=node_emb_size,
            hidden_layer_sizes=[mlp1_hidden_dim] * mlp1_depth,
            out_features=f_conn_elems,
            dropout_p=mlp1_dropout_p,
        )

        # Tier 2 — graph-conditioned final distributions
        self.fAddNet2 = MLP(
            in_features=(max_n_nodes * f_add_elems + graph_emb_size),
            hidden_layer_sizes=[mlp2_hidden_dim] * mlp2_depth,
            out_features=f_add_elems * max_n_nodes,
            dropout_p=mlp2_dropout_p,
        )
        self.fConnNet2 = MLP(
            in_features=(max_n_nodes * f_conn_elems + graph_emb_size),
            hidden_layer_sizes=[mlp2_hidden_dim] * mlp2_depth,
            out_features=f_conn_elems * max_n_nodes,
            dropout_p=mlp2_dropout_p,
        )
        self.fTermNet2 = MLP(
            in_features=graph_emb_size,
            hidden_layer_sizes=[mlp2_hidden_dim] * mlp2_depth,
            out_features=f_term_elems,
            dropout_p=mlp2_dropout_p,
        )

    def forward(
        self, node_level_output: torch.Tensor, graph_embedding_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            node_level_output:    Hidden states for all nodes in the batch.
                                  Shape: (batch, max_n_nodes, hidden_node_features)
            graph_embedding_batch: Graph-level embeddings from AttentionReadout.
                                  Shape: (batch, graph_emb_size)

        Returns:
            Flat, unnormalised action probabilities logits.
            Shape: (batch, len_f_add + len_f_conn + 1)
        """
        # Tier-1: per-node preliminary f_add and f_conn
        f_add_1 = self.fAddNet1(node_level_output)
        f_conn_1 = self.fConnNet1(node_level_output)

        # Flatten (batch, nodes, features) → (batch, nodes * features)
        f_add_1 = f_add_1.view(f_add_1.size(0), f_add_1.size(1) * f_add_1.size(2))
        f_conn_1 = f_conn_1.view(f_conn_1.size(0), f_conn_1.size(1) * f_conn_1.size(2))

        # Tier-2: condition on graph embedding to produce final action probabilities components
        f_add_2 = self.fAddNet2(
            torch.cat((f_add_1, graph_embedding_batch), dim=1).unsqueeze(dim=1)
        )
        f_conn_2 = self.fConnNet2(
            torch.cat((f_conn_1, graph_embedding_batch), dim=1).unsqueeze(dim=1)
        )
        f_term_2 = self.fTermNet2(graph_embedding_batch)

        # Flatten and concatenate: [f_add | f_conn | f_term]
        return torch.cat(
            (f_add_2.squeeze(dim=1), f_conn_2.squeeze(dim=1), f_term_2), dim=1
        )
