"""
Defines the `SummationMPNN` base class for message-passing neural networks.
"""
from collections import namedtuple
import torch


class SummationMPNN(torch.nn.Module):
    """
    Abstract base class for summation-based message-passing neural networks (MPNNs).

    In this framework, each node collects messages from its neighbours, sums them,
    and uses a gating mechanism (GRU) to update its own hidden state.  After a
    fixed number of message-passing rounds the hidden states are pooled into a
    single graph-level embedding and passed to a readout network that predicts the
    Action Probability Distribution (APD) — the probability of each next
    construction step (add a node, connect two nodes, or terminate the graph).

    Concrete model classes (e.g. `GGNN`) inherit from this class and implement
    `message_terms`, `update`, and `readout`.
    """
    def __init__(self, constants: namedtuple):
        super().__init__()

        self.hidden_node_features = constants.hidden_node_features
        self.edge_features        = constants.n_edge_features
        self.message_size         = constants.message_size
        self.message_passes       = constants.message_passes
        self.constants            = constants

    def message_terms(self, nodes: torch.Tensor, node_neighbours: torch.Tensor,
                      edges: torch.Tensor) -> torch.Tensor:
        """
        Computes the outgoing message from each (node, neighbour) edge pair.

        Must be implemented by every subclass.

        Args:
            nodes:           Hidden states of the source nodes.
                             Shape: (total edges in batch, hidden_node_features)
            node_neighbours: Hidden states of the neighbouring nodes.
                             Shape: (total edges in batch, hidden_node_features)
            edges:           One-hot bond-type features for each edge.
                             Shape: (total edges in batch, n_edge_features)

        Returns:
            messages: One message vector per edge.
                      Shape: (total edges in batch, message_size)
        """
        raise NotImplementedError

    def update(self, nodes: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """
        Updates each node's hidden state given the aggregated incoming messages.

        Must be implemented by every subclass.

        Args:
            nodes:    Current hidden states.
                      Shape: (total nodes with at least one neighbour, hidden_node_features)
            messages: Summed incoming messages for each node.
                      Shape: (total nodes with at least one neighbour, message_size)

        Returns:
            Updated hidden states. Same shape as `nodes`.
        """
        raise NotImplementedError

    def readout(self, hidden_nodes: torch.Tensor, input_nodes: torch.Tensor,
                node_mask: torch.Tensor) -> torch.Tensor:
        """
        Produces the APD prediction from the final node hidden states.

        Must be implemented by every subclass.

        Args:
            hidden_nodes: Final hidden node states after all message-passing rounds.
                          Shape: (batch, max_n_nodes, hidden_node_features)
            input_nodes:  Original (input) node feature vectors.
                          Shape: (batch, max_n_nodes, n_node_features)
            node_mask:    Boolean mask; True for real nodes, False for padding.
                          Shape: (batch, max_n_nodes)

        Returns:
            apd: Flat, unnormalised APD logits (softmax is applied externally).
                 Shape: (batch, len_f_add + len_f_conn + 1)
        """
        raise NotImplementedError

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """
        Runs the full message-passing loop and returns the APD logits.

        Steps:
          1. Build a sparse representation of all (batch, node, neighbour) triples
             from the adjacency matrix.
          2. Run `message_passes` rounds of: compute messages → sum messages per node
             → update hidden state with GRU.
          3. Call `readout` on the final hidden states to produce the APD.

        Args:
            nodes: Node feature matrices, one per graph.
                   Shape: (batch, max_n_nodes, n_node_features)
            edges: Edge feature tensors, one per graph.
                   Shape: (batch, max_n_nodes, max_n_nodes, n_edge_features)

        Returns:
            apd: Flat, unnormalised APD logits.
                 Shape: (batch, len_f_add + len_f_conn + 1)
        """
        adjacency = torch.sum(edges, dim=3)

        # Collect all non-zero (batch, src_node, dst_node) index triples
        (edge_batch_batch_idc,
         edge_batch_node_idc,
         edge_batch_nghb_idc) = adjacency.nonzero(as_tuple=True)

        # Collect all (batch, node) pairs that have at least one neighbour
        (node_batch_batch_idc, node_batch_node_idc) = adjacency.sum(-1).nonzero(as_tuple=True)

        # message_summation_matrix[i, j] = 1 iff edge j is incident to node i
        same_batch = node_batch_batch_idc.view(-1, 1) == edge_batch_batch_idc
        same_node  = node_batch_node_idc.view(-1, 1)  == edge_batch_node_idc
        message_summation_matrix = (same_batch * same_node).float()

        edge_batch_edges = edges[edge_batch_batch_idc, edge_batch_node_idc, edge_batch_nghb_idc, :]

        # Initialise hidden states from input node features (zero-padded to hidden dim)
        hidden_nodes = torch.zeros(nodes.shape[0], nodes.shape[1],
                                   self.hidden_node_features,
                                   device=self.constants.device)
        hidden_nodes[:nodes.shape[0], :nodes.shape[1], :nodes.shape[2]] = nodes.clone()
        node_batch_nodes = hidden_nodes[node_batch_batch_idc, node_batch_node_idc, :]

        for _ in range(self.message_passes):
            edge_batch_nodes = hidden_nodes[edge_batch_batch_idc, edge_batch_node_idc, :]
            edge_batch_nghbs  = hidden_nodes[edge_batch_batch_idc, edge_batch_nghb_idc, :]

            message_terms = self.message_terms(edge_batch_nodes, edge_batch_nghbs, edge_batch_edges)
            if len(message_terms.size()) == 1:
                message_terms = message_terms.unsqueeze(0)

            messages = torch.matmul(message_summation_matrix, message_terms)

            node_batch_nodes = self.update(node_batch_nodes, messages)
            hidden_nodes[node_batch_batch_idc, node_batch_node_idc, :] = node_batch_nodes.clone()

        node_mask = adjacency.sum(-1) != 0
        return self.readout(hidden_nodes, nodes, node_mask)
