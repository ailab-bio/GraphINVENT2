"""
ConditionEncoder: maps a property (condition) vector to a fixed-size embedding
used to initialise the virtual seed node in the GGNN forward pass.
"""

import torch


class ConditionEncoder(torch.nn.Module):
    """
    Two-layer MLP that encodes a conditioning property vector.

    The resulting embedding is used to initialise the hidden state of a virtual
    seed node that is prepended to every molecular graph before message passing.
    All real nodes receive a message from the seed at every message-passing
    round, injecting the conditioning signal into the graph representation.

    Args:
        condition_dim:           Dimensionality of the input property vector.
        hidden_dim:              Width of the hidden layer.
        condition_embedding_dim: Dimensionality of the output embedding.
    """

    def __init__(
        self,
        condition_dim: int,
        hidden_dim: int,
        condition_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(condition_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, condition_embedding_dim),
        )

    def forward(self, condition_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            condition_vector: Shape (batch, condition_dim).

        Returns:
            condition_embedding: Shape (batch, condition_embedding_dim).
        """
        return self.net(condition_vector)
