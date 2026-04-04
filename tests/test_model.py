"""
Unit tests for GraphINVENT2 model components.

Covers five categories:
  1. Shape and dimension checks    – GNN forward pass, constant consistency
  2. Action probability properties – sums to 1, finite, non-negative
  3. Autoregressive step tests     – partial graphs → valid distributions
  4. Reproducibility               – same seed → identical output
  5. Training sanity checks        – gradient flow, loss decreases
  6. Edge cases / boundary         – empty graph, fully-connected, max-size
  7. GraphGenerator integration    – count, shapes, termination flags

GraphGenerator reads a module-level ``constants`` singleton imported from
``parameters.constants``.  These tests monkey-patch that binding with a small,
self-contained test constants namedtuple so no real dataset is required.

Usage:
    pytest tests/test_model.py -v
"""

import random
import sys
from collections import namedtuple
from math import prod
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from rdkit.Chem.rdchem import BondType

# ---------------------------------------------------------------------------
# Path setup – make graphinvent/ importable
# ---------------------------------------------------------------------------

_GRAPHINVENT = Path(__file__).resolve().parent.parent / "src/graphinvent"
if str(_GRAPHINVENT) not in sys.path:
    sys.path.insert(0, str(_GRAPHINVENT))

import gnn.mpnn as mpnn_module
import GraphGenerator as gg_module
from GraphGenerator import GraphGenerator
from MolecularGraph import GenerationGraph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_constants(
    max_n_nodes: int = 5,
    batch_size: int = 4,
    n_samples: int = 8,
    device: str = "cpu",
) -> namedtuple:
    """
    Build a minimal constants namedtuple for fast unit tests.

    Uses a tiny vocabulary (C/N/O, neutral charge, no chirality, no aromatic
    bonds) and a tiny GGNN (hidden dim 16) so each test completes quickly.
    """
    atom_types = ["C", "N", "O"]
    formal_charge = [0]
    imp_H = [0, 1, 2, 3]
    chirality = ["None"]

    n_atom_types = len(atom_types)  # 3
    n_formal_charge = len(formal_charge)  # 1
    n_imp_H = len(imp_H)  # 4
    n_chirality = 0  # use_chirality = False
    n_node_features = n_atom_types + n_formal_charge + n_imp_H  # 8
    n_edge_features = 3  # SINGLE / DOUBLE / TRIPLE

    bondtype_to_int = {
        BondType.SINGLE: 0,
        BondType.DOUBLE: 1,
        BondType.TRIPLE: 2,
    }
    int_to_bondtype = {v: k for k, v in bondtype_to_int.items()}

    # f_add: max_n_nodes × n_atom_types × n_formal_charge × n_imp_H × n_edge_features
    len_f_add_per_node = n_atom_types * n_formal_charge * n_imp_H * n_edge_features
    len_f_add = max_n_nodes * len_f_add_per_node
    len_f_conn_per_node = n_edge_features
    len_f_conn = max_n_nodes * len_f_conn_per_node
    dim_action_probs = len_f_add + len_f_conn + 1

    dim_f_add = [max_n_nodes, n_atom_types, n_formal_charge, n_imp_H, n_edge_features]
    dim_f_conn = [max_n_nodes, n_edge_features]
    dim_f_term = [1]
    dim_nodes = [max_n_nodes, n_node_features]
    dim_edges = [max_n_nodes, max_n_nodes, n_edge_features]

    h = 16  # hidden dimension for all MLP layers (tiny for speed)

    fields = [
        "atom_types",
        "formal_charge",
        "imp_H",
        "chirality",
        "n_atom_types",
        "n_formal_charge",
        "n_imp_H",
        "n_chirality",
        "n_node_features",
        "n_edge_features",
        "bondtype_to_int",
        "int_to_bondtype",
        "max_n_nodes",
        "device",
        "len_f_add",
        "len_f_conn",
        "len_f_add_per_node",
        "len_f_conn_per_node",
        "dim_f_add",
        "dim_f_conn",
        "dim_f_term",
        "dim_action_probs",
        "dim_nodes",
        "dim_edges",
        "hidden_node_features",
        "message_size",
        "message_passes",
        "enn_depth",
        "enn_hidden_dim",
        "enn_dropout_p",
        "mlp1_depth",
        "mlp1_hidden_dim",
        "mlp1_dropout_p",
        "mlp2_depth",
        "mlp2_hidden_dim",
        "mlp2_dropout_p",
        "gather_width",
        "gather_att_depth",
        "gather_att_hidden_dim",
        "gather_att_dropout_p",
        "gather_emb_depth",
        "gather_emb_hidden_dim",
        "gather_emb_dropout_p",
        "use_chirality",
        "use_explicit_H",
        "ignore_H",
        "use_aromatic_bonds",
        "n_samples",
        "batch_size",
        "big_positive",
        "big_negative",
        "job_dir",
        "decoding_route",
    ]
    C = namedtuple("Constants", fields)

    return C(
        atom_types=atom_types,
        formal_charge=formal_charge,
        imp_H=imp_H,
        chirality=chirality,
        n_atom_types=n_atom_types,
        n_formal_charge=n_formal_charge,
        n_imp_H=n_imp_H,
        n_chirality=n_chirality,
        n_node_features=n_node_features,
        n_edge_features=n_edge_features,
        bondtype_to_int=bondtype_to_int,
        int_to_bondtype=int_to_bondtype,
        max_n_nodes=max_n_nodes,
        device=device,
        len_f_add=len_f_add,
        len_f_conn=len_f_conn,
        len_f_add_per_node=len_f_add_per_node,
        len_f_conn_per_node=len_f_conn_per_node,
        dim_f_add=dim_f_add,
        dim_f_conn=dim_f_conn,
        dim_f_term=dim_f_term,
        dim_action_probs=dim_action_probs,
        dim_nodes=dim_nodes,
        dim_edges=dim_edges,
        hidden_node_features=h,
        message_size=h,
        message_passes=2,
        enn_depth=2,
        enn_hidden_dim=h,
        enn_dropout_p=0.0,
        mlp1_depth=2,
        mlp1_hidden_dim=h,
        mlp1_dropout_p=0.0,
        mlp2_depth=2,
        mlp2_hidden_dim=h,
        mlp2_dropout_p=0.0,
        gather_width=h,
        gather_att_depth=2,
        gather_att_hidden_dim=h,
        gather_att_dropout_p=0.0,
        gather_emb_depth=2,
        gather_emb_hidden_dim=h,
        gather_emb_dropout_p=0.0,
        use_chirality=False,
        use_explicit_H=False,
        ignore_H=False,
        use_aromatic_bonds=False,
        n_samples=n_samples,
        batch_size=batch_size,
        big_positive=1e6,
        big_negative=-1e6,
        job_dir="/tmp/graphinvent_test/",
        decoding_route="bfs",
    )


def _zero_batch(
    constants: namedtuple,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all-zero (empty) node and edge tensors."""
    N, NF = constants.max_n_nodes, constants.n_node_features
    EF = constants.n_edge_features
    return (
        torch.zeros(batch_size, N, NF),
        torch.zeros(batch_size, N, N, EF),
    )


def _one_node_batch(
    constants: namedtuple,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return node/edge tensors where every graph has exactly one C atom."""
    nodes, edges = _zero_batch(constants, batch_size)
    nodes[:, 0, 0] = 1.0  # atom_type[0] = 'C', one-hot at feature 0
    return nodes, edges


def _random_target_apd(constants: namedtuple, batch_size: int) -> torch.Tensor:
    """Return a random valid action probability distribution (sums to 1)."""
    raw = torch.rand(batch_size, constants.dim_action_probs)
    return F.softmax(raw, dim=-1)


# ---------------------------------------------------------------------------
# Module-level fixtures (shared across all test classes)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def constants() -> namedtuple:
    return _make_constants()


@pytest.fixture(scope="module")
def model(constants: namedtuple) -> torch.nn.Module:
    _set_all_seeds(0)
    m = mpnn_module.GGNN(constants)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# 1. Shape and dimension tests
# ---------------------------------------------------------------------------


class TestGNNShapes:
    """Verify that all tensor shapes are consistent with the constants."""

    def test_output_shape_empty_graphs(self, model, constants):
        """Forward on all-zero (empty) graphs returns correct logit shape."""
        B = 3
        nodes, edges = _zero_batch(constants, B)
        with torch.no_grad():
            out = model(nodes, edges)
        assert out.shape == (B, constants.dim_action_probs)

    def test_output_shape_partial_graphs(self, model, constants):
        """Forward on one-node graphs returns correct logit shape."""
        B = 4
        nodes, edges = _one_node_batch(constants, B)
        with torch.no_grad():
            out = model(nodes, edges)
        assert out.shape == (B, constants.dim_action_probs)

    def test_output_shape_batch_size_one(self, model, constants):
        """Batch of 1 still produces a correctly shaped output."""
        nodes, edges = _zero_batch(constants, 1)
        with torch.no_grad():
            out = model(nodes, edges)
        assert out.shape == (1, constants.dim_action_probs)

    def test_output_shape_larger_batch(self, model, constants):
        """Output shape scales linearly with batch size."""
        B = 16
        nodes, edges = _one_node_batch(constants, B)
        with torch.no_grad():
            out = model(nodes, edges)
        assert out.shape == (B, constants.dim_action_probs)

    def test_dim_action_probs_decomposition(self, constants):
        """dim_action_probs must equal len_f_add + len_f_conn + 1."""
        assert (
            constants.dim_action_probs == constants.len_f_add + constants.len_f_conn + 1
        )

    def test_len_f_add_matches_dim_product(self, constants):
        """len_f_add must equal the product of dim_f_add."""
        assert constants.len_f_add == prod(constants.dim_f_add)

    def test_len_f_conn_matches_dim_product(self, constants):
        """len_f_conn must equal the product of dim_f_conn."""
        assert constants.len_f_conn == prod(constants.dim_f_conn)

    def test_n_node_features_decomposition(self, constants):
        """n_node_features must equal sum of per-feature vocab sizes."""
        # Without chirality or explicit H: atom_types + formal_charge + imp_H
        expected = (
            constants.n_atom_types + constants.n_formal_charge + constants.n_imp_H
        )
        assert constants.n_node_features == expected

    def test_f_add_segment_width(self, model, constants):
        """The f_add slice of the output has length len_f_add."""
        nodes, edges = _one_node_batch(constants, 2)
        with torch.no_grad():
            logits = model(nodes, edges)
        assert logits[:, : constants.len_f_add].shape == (2, constants.len_f_add)

    def test_f_conn_segment_width(self, model, constants):
        """The f_conn slice of the output has length len_f_conn."""
        nodes, edges = _one_node_batch(constants, 2)
        with torch.no_grad():
            logits = model(nodes, edges)
        f_conn = logits[:, constants.len_f_add : -1]
        assert f_conn.shape == (2, constants.len_f_conn)

    def test_f_term_is_scalar_per_graph(self, model, constants):
        """The f_term element is a single scalar per graph in the batch."""
        nodes, edges = _one_node_batch(constants, 3)
        with torch.no_grad():
            logits = model(nodes, edges)
        assert logits[:, -1].shape == (3,)


# ---------------------------------------------------------------------------
# 2. Action probability properties
# ---------------------------------------------------------------------------


class TestActionProbProperties:
    """Verify statistical properties of the softmax output."""

    def test_softmax_sums_to_one(self, model, constants):
        """After softmax, action probs for every graph sum to 1."""
        B = 6
        nodes, edges = _one_node_batch(constants, B)
        with torch.no_grad():
            probs = F.softmax(model(nodes, edges), dim=-1)
        assert torch.allclose(
            probs.sum(dim=-1), torch.ones(B), atol=1e-5
        ), f"Row sums: {probs.sum(dim=-1)}"

    def test_softmax_nonnegative(self, model, constants):
        """After softmax, all probabilities are >= 0."""
        nodes, edges = _zero_batch(constants, 4)
        with torch.no_grad():
            probs = F.softmax(model(nodes, edges), dim=-1)
        assert (probs >= 0).all()

    def test_logits_are_finite(self, model, constants):
        """Model output logits must not contain NaN or Inf."""
        nodes, edges = _one_node_batch(constants, 5)
        with torch.no_grad():
            out = model(nodes, edges)
        assert torch.isfinite(out).all(), "Logits contain NaN or Inf"

    def test_logits_vary_with_input(self, model, constants):
        """Different graph states must produce different logit vectors."""
        nodes_empty, edges_empty = _zero_batch(constants, 1)
        nodes_one, edges_one = _one_node_batch(constants, 1)
        with torch.no_grad():
            out_empty = model(nodes_empty, edges_empty)
            out_one = model(nodes_one, edges_one)
        assert not torch.allclose(out_empty, out_one), (
            "Model output is identical for empty vs. one-node graph — "
            "model may be ignoring its input"
        )

    def test_output_length_matches_constant(self, model, constants):
        """Output dimension must equal dim_action_probs stored in constants."""
        nodes, edges = _zero_batch(constants, 2)
        with torch.no_grad():
            out = model(nodes, edges)
        assert out.shape[-1] == constants.dim_action_probs


# ---------------------------------------------------------------------------
# 3. Autoregressive step tests
# ---------------------------------------------------------------------------


class TestAutoregressiveStep:
    """
    Given a partial graph state, the model should produce a valid probability
    distribution over next actions.
    """

    def test_empty_graph_valid_distribution(self, model, constants):
        """Model on a 0-node graph yields a valid probability distribution."""
        nodes, edges = _zero_batch(constants, 1)
        with torch.no_grad():
            probs = F.softmax(model(nodes, edges), dim=-1)
        assert torch.isfinite(probs).all()
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)

    def test_one_node_graph_valid_distribution(self, model, constants):
        """Model on a 1-node graph yields a valid probability distribution."""
        nodes, edges = _one_node_batch(constants, 1)
        with torch.no_grad():
            probs = F.softmax(model(nodes, edges), dim=-1)
        assert torch.isfinite(probs).all()
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)

    def test_two_node_bonded_graph_valid_distribution(self, model, constants):
        """Model on a 2-node graph (C–N, single bond) gives a valid distribution."""
        B, N, NF, EF = (
            1,
            constants.max_n_nodes,
            constants.n_node_features,
            constants.n_edge_features,
        )
        nodes = torch.zeros(B, N, NF)
        edges = torch.zeros(B, N, N, EF)
        nodes[0, 0, 0] = 1.0  # C at node 0
        nodes[0, 1, 1] = 1.0  # N at node 1
        edges[0, 0, 1, 0] = 1.0  # single bond 0→1
        edges[0, 1, 0, 0] = 1.0  # single bond 1→0 (symmetric)

        with torch.no_grad():
            probs = F.softmax(model(nodes, edges), dim=-1)

        assert torch.isfinite(probs).all()
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B), atol=1e-5)

    def test_adding_node_changes_distribution(self, model, constants):
        """Adding an atom to the graph must change the predicted distribution."""
        B, N, NF, EF = (
            1,
            constants.max_n_nodes,
            constants.n_node_features,
            constants.n_edge_features,
        )
        nodes0 = torch.zeros(B, N, NF)
        edges0 = torch.zeros(B, N, N, EF)
        nodes1 = nodes0.clone()
        nodes1[0, 0, 0] = 1.0  # add one carbon

        with torch.no_grad():
            probs0 = F.softmax(model(nodes0, edges0), dim=-1)
            probs1 = F.softmax(model(nodes1, edges0), dim=-1)

        assert not torch.allclose(probs0, probs1, atol=1e-4), (
            "Distribution is identical before and after adding an atom — "
            "the model may not be conditioning on graph state"
        )

    def test_batch_produces_per_graph_distributions(self, model, constants):
        """A batch of different graph states yields distinct per-graph distributions."""
        B, N, NF, EF = (
            4,
            constants.max_n_nodes,
            constants.n_node_features,
            constants.n_edge_features,
        )
        nodes = torch.zeros(B, N, NF)
        edges = torch.zeros(B, N, N, EF)
        # Give each graph a different number of populated nodes
        for b in range(B):
            for i in range(b):  # graph b has b nodes
                nodes[b, i, b % constants.n_atom_types] = 1.0

        with torch.no_grad():
            probs = F.softmax(model(nodes, edges), dim=-1)

        # Each row sums to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B), atol=1e-5)
        # Rows should not all be identical
        are_same = all(
            torch.allclose(probs[0], probs[b], atol=1e-4) for b in range(1, B)
        )
        assert not are_same, "All graphs in the batch produced the same distribution"


# ---------------------------------------------------------------------------
# 4. Reproducibility tests
# ---------------------------------------------------------------------------


class TestReproducibility:

    def test_same_init_seed_same_logits(self, constants):
        """Two models initialised with the same seed produce identical logits."""
        nodes, edges = _one_node_batch(constants, 4)

        _set_all_seeds(99)
        m1 = mpnn_module.GGNN(constants)
        m1.eval()
        with torch.no_grad():
            out1 = m1(nodes, edges)

        _set_all_seeds(99)
        m2 = mpnn_module.GGNN(constants)
        m2.eval()
        with torch.no_grad():
            out2 = m2(nodes, edges)

        assert torch.allclose(
            out1, out2
        ), "Same init seed should yield identical logits"

    def test_different_init_seeds_different_logits(self, constants):
        """Two models initialised with different seeds produce different logits."""
        nodes, edges = _one_node_batch(constants, 4)

        _set_all_seeds(1)
        m1 = mpnn_module.GGNN(constants)
        m1.eval()
        with torch.no_grad():
            out1 = m1(nodes, edges)

        _set_all_seeds(2)
        m2 = mpnn_module.GGNN(constants)
        m2.eval()
        with torch.no_grad():
            out2 = m2(nodes, edges)

        assert not torch.allclose(
            out1, out2
        ), "Different init seeds produced identical logits — weight init may be broken"

    def test_eval_mode_is_deterministic(self, constants):
        """The same model in eval mode produces identical output on repeated calls."""
        _set_all_seeds(7)
        m = mpnn_module.GGNN(constants)
        m.eval()
        nodes, edges = _one_node_batch(constants, 3)

        with torch.no_grad():
            out1 = m(nodes, edges)
            out2 = m(nodes, edges)

        assert torch.allclose(out1, out2), "Eval-mode forward is not deterministic"


# ---------------------------------------------------------------------------
# 5. Training sanity checks
# ---------------------------------------------------------------------------


class TestTrainingSanity:
    """Verify that the model can receive gradients and learn."""

    def test_gradients_flow_through_all_parameters(self, constants):
        """All parameters must receive finite gradients after one backward pass.

        The virtual-edge MLP (``msg_nns[n_edge_features]``) is only activated
        when conditioning is used; it is intentionally skipped here because
        unconditional forward passes never create virtual edges.
        """
        _set_all_seeds(0)
        m = mpnn_module.GGNN(constants)
        m.train()
        optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)

        B = 4
        nodes, edges = _one_node_batch(constants, B)
        target = _random_target_apd(constants, B)

        optimizer.zero_grad()
        logits = m(nodes, edges)
        loss = F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")
        loss.backward()

        # The last msg_nn (index n_edge_features) handles the virtual edge type
        # used only during conditional generation — skip it for unconditional tests.
        virtual_mlp_prefix = f"msg_nns.{constants.n_edge_features}."

        for name, p in m.named_parameters():
            if name.startswith(virtual_mlp_prefix):
                continue  # virtual-edge MLP unused in unconditional forward
            if name.startswith("condition_encoder."):
                continue  # condition encoder unused when condition_dim == 0
            assert p.grad is not None, f"Parameter has no gradient: {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient for: {name}"

    def test_loss_is_finite_and_nonnegative(self, constants):
        """KL-divergence loss must be finite and non-negative on a valid batch."""
        _set_all_seeds(1)
        m = mpnn_module.GGNN(constants)
        m.train()

        B = 4
        nodes, edges = _one_node_batch(constants, B)
        target = _random_target_apd(constants, B)

        logits = m(nodes, edges)
        loss = F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        assert loss.item() >= 0.0, f"KL divergence cannot be negative: {loss.item()}"

    def test_gradient_step_moves_parameters(self, constants):
        """At least one parameter must change value after a gradient step."""
        _set_all_seeds(3)
        m = mpnn_module.GGNN(constants)
        m.train()
        optimizer = torch.optim.Adam(m.parameters(), lr=1e-2)

        params_before = {n: p.detach().clone() for n, p in m.named_parameters()}

        B = 4
        nodes, edges = _one_node_batch(constants, B)
        target = _random_target_apd(constants, B)

        optimizer.zero_grad()
        loss = F.kl_div(
            F.log_softmax(m(nodes, edges), dim=-1), target, reduction="batchmean"
        )
        loss.backward()
        optimizer.step()

        changed = any(
            not torch.allclose(params_before[n], p.detach())
            for n, p in m.named_parameters()
        )
        assert changed, "No parameter changed after a gradient step"

    def test_model_overfits_single_example(self, constants):
        """
        After many gradient steps on a fixed single example, the loss should
        drop substantially.  If it cannot memorise one example something is
        fundamentally broken.
        """
        _set_all_seeds(42)
        m = mpnn_module.GGNN(constants)
        m.train()
        optimizer = torch.optim.Adam(m.parameters(), lr=1e-2)

        B = 1
        nodes, edges = _one_node_batch(constants, B)
        # Target: always choose the terminate action (last element)
        target = torch.zeros(B, constants.dim_action_probs)
        target[:, -1] = 1.0

        def _loss():
            return F.kl_div(
                F.log_softmax(m(nodes, edges), dim=-1), target, reduction="batchmean"
            )

        loss_0 = _loss().item()

        for _ in range(150):
            optimizer.zero_grad()
            _loss().backward()
            optimizer.step()

        loss_final = _loss().item()
        assert loss_final < loss_0 * 0.5, (
            f"Model failed to overfit: initial loss {loss_0:.4f}, "
            f"final loss {loss_final:.4f} — expected at least 50% reduction"
        )


# ---------------------------------------------------------------------------
# 6. Edge cases and boundary conditions
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_all_zero_input_produces_finite_output(self, constants):
        """All-zero input (empty graph) must produce finite logits."""
        _set_all_seeds(0)
        m = mpnn_module.GGNN(constants)
        m.eval()
        nodes, edges = _zero_batch(constants, 2)
        with torch.no_grad():
            out = m(nodes, edges)
        assert torch.isfinite(out).all(), "Inf/NaN from empty-graph input"
        assert out.shape == (2, constants.dim_action_probs)

    def test_single_isolated_atom(self, constants):
        """Single atom with no bonds must produce a valid distribution."""
        _set_all_seeds(0)
        m = mpnn_module.GGNN(constants)
        m.eval()
        B, N, NF, EF = (
            1,
            constants.max_n_nodes,
            constants.n_node_features,
            constants.n_edge_features,
        )
        nodes = torch.zeros(B, N, NF)
        edges = torch.zeros(B, N, N, EF)
        nodes[0, 0, 0] = 1.0  # one carbon, no bonds

        with torch.no_grad():
            probs = F.softmax(m(nodes, edges), dim=-1)

        assert torch.isfinite(probs).all()
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B), atol=1e-5)

    def test_fully_occupied_graph(self, constants):
        """Graph with max_n_nodes atoms fully connected must not crash."""
        _set_all_seeds(0)
        m = mpnn_module.GGNN(constants)
        m.eval()
        B, N, NF, EF = (
            1,
            constants.max_n_nodes,
            constants.n_node_features,
            constants.n_edge_features,
        )
        nodes = torch.zeros(B, N, NF)
        edges = torch.zeros(B, N, N, EF)
        for i in range(N):
            nodes[0, i, 0] = 1.0  # all carbons
        for i in range(N):
            for j in range(N):
                if i != j:
                    edges[0, i, j, 0] = 1.0  # single bonds everywhere

        with torch.no_grad():
            out = m(nodes, edges)

        assert out.shape == (B, constants.dim_action_probs)
        assert torch.isfinite(out).all()

    def test_output_changes_when_bond_added(self, constants):
        """Adding a bond between two existing atoms must change the output."""
        _set_all_seeds(0)
        m = mpnn_module.GGNN(constants)
        m.eval()
        B, N, NF, EF = (
            1,
            constants.max_n_nodes,
            constants.n_node_features,
            constants.n_edge_features,
        )
        nodes = torch.zeros(B, N, NF)
        edges_no_bond = torch.zeros(B, N, N, EF)
        edges_with_bond = torch.zeros(B, N, N, EF)

        nodes[0, 0, 0] = 1.0  # C at 0
        nodes[0, 1, 1] = 1.0  # N at 1
        edges_with_bond[0, 0, 1, 0] = 1.0
        edges_with_bond[0, 1, 0, 0] = 1.0

        with torch.no_grad():
            out_no_bond = m(nodes, edges_no_bond)
            out_with_bond = m(nodes, edges_with_bond)

        assert not torch.allclose(
            out_no_bond, out_with_bond, atol=1e-6
        ), "Adding a bond did not change model output"

    def test_linear_chain_max_nodes(self, constants):
        """Linear chain of max_n_nodes atoms must produce finite, correct-shape output."""
        _set_all_seeds(0)
        m = mpnn_module.GGNN(constants)
        m.eval()
        B, N, NF, EF = (
            1,
            constants.max_n_nodes,
            constants.n_node_features,
            constants.n_edge_features,
        )
        nodes = torch.zeros(B, N, NF)
        edges = torch.zeros(B, N, N, EF)
        for i in range(N):
            nodes[0, i, 0] = 1.0
        for i in range(N - 1):
            edges[0, i, i + 1, 0] = 1.0
            edges[0, i + 1, i, 0] = 1.0

        with torch.no_grad():
            out = m(nodes, edges)

        assert out.shape == (B, constants.dim_action_probs)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 7. GraphGenerator integration tests
# ---------------------------------------------------------------------------


class TestGraphGenerator:
    """
    End-to-end tests for the autoregressive generation loop.

    GraphGenerator reads a module-level ``constants`` singleton; each test
    monkey-patches that binding with the test constants namedtuple so no real
    dataset or environment is required.
    """

    @pytest.fixture(autouse=True)
    def patch_constants(self, monkeypatch):
        """Replace the global constants used by GraphGenerator with test constants."""
        tc = _make_constants(max_n_nodes=5, batch_size=4, n_samples=4)
        monkeypatch.setattr(gg_module, "constants", tc)
        self._tc = tc

    @pytest.fixture()
    def gen_model(self):
        _set_all_seeds(0)
        m = mpnn_module.GGNN(self._tc)
        m.eval()
        return m

    def test_returns_correct_number_of_graphs(self, gen_model):
        """sample() must return exactly batch_size GenerationGraph objects."""
        gen = GraphGenerator(gen_model, self._tc.batch_size)
        with torch.no_grad():
            graphs, _, _, _ = gen.sample()
        assert len(graphs) == self._tc.batch_size

    def test_returned_graphs_are_correct_type(self, gen_model):
        """Every returned graph must be a GenerationGraph instance."""
        gen = GraphGenerator(gen_model, self._tc.batch_size)
        with torch.no_grad():
            graphs, _, _, _ = gen.sample()
        for g in graphs:
            assert isinstance(
                g, GenerationGraph
            ), f"Expected GenerationGraph, got {type(g)}"

    def test_final_loglikelihoods_shape(self, gen_model):
        """final_loglikelihoods must have one entry per generated molecule."""
        gen = GraphGenerator(gen_model, self._tc.batch_size)
        with torch.no_grad():
            _, _, final_ll, _ = gen.sample()
        assert final_ll.shape[0] == self._tc.batch_size

    def test_final_loglikelihoods_nonpositive(self, gen_model):
        """Log-likelihoods must be <= 0 (log of probabilities in (0, 1])."""
        gen = GraphGenerator(gen_model, self._tc.batch_size)
        with torch.no_grad():
            _, _, final_ll, _ = gen.sample()
        assert (
            final_ll <= 0
        ).all(), f"Positive log-likelihoods found: {final_ll[final_ll > 0]}"

    def test_properly_terminated_is_binary(self, gen_model):
        """properly_terminated tensor must contain only 0s and 1s."""
        gen = GraphGenerator(gen_model, self._tc.batch_size)
        with torch.no_grad():
            _, _, _, pt = gen.sample()
        assert pt.shape[0] == self._tc.batch_size
        unique_vals = set(pt.tolist())
        assert unique_vals.issubset(
            {0, 1}
        ), f"Non-binary values in properly_terminated: {unique_vals}"

    def test_reproducible_with_same_seed(self):
        """Same model init seed + same sampling seed → identical log-likelihoods."""
        tc = self._tc

        def _run(seed):
            _set_all_seeds(seed)
            m = mpnn_module.GGNN(tc)
            m.eval()
            gen = GraphGenerator(m, tc.batch_size)
            with torch.no_grad():
                _, _, final_ll, pt = gen.sample()
            return final_ll, pt

        ll1, pt1 = _run(seed=77)
        ll2, pt2 = _run(seed=77)

        assert torch.allclose(ll1, ll2), "Log-likelihoods differ despite identical seed"
        assert torch.equal(
            pt1, pt2
        ), "properly_terminated differs despite identical seed"

    def test_generation_terminates(self, gen_model):
        """sample() must complete and return (no infinite loop)."""
        gen = GraphGenerator(gen_model, self._tc.batch_size)
        with torch.no_grad():
            result = gen.sample()
        assert result is not None
        assert len(result) == 4  # (graphs, likelihoods, final_ll, properly_terminated)
