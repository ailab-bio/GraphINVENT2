"""
Conditional tensor shapes and gradients.

A condition signal that never reaches the output looks exactly like a model
that failed to learn, so these tests assert that two different condition
vectors give different logits and that the parameters carrying the condition
receive gradient.
"""

import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest
import torch
from rdkit.Chem.rdchem import BondType

_GRAPHINVENT = Path(__file__).resolve().parent.parent / "src/graphinvent"
if str(_GRAPHINVENT) not in sys.path:
    sys.path.insert(0, str(_GRAPHINVENT))

MAX_N_NODES = 6


def _constants(condition_dim: int):
    atom_types = ["C", "N", "O"]
    formal_charge = [0]
    imp_H = [0, 1, 2, 3]
    bondtype_to_int = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2}

    n_atom_types = len(atom_types)
    n_formal_charge = len(formal_charge)
    n_imp_H = len(imp_H)
    n_edge_features = len(bondtype_to_int)
    n_node_features = n_atom_types + n_formal_charge + n_imp_H

    dim_f_add = [
        MAX_N_NODES,
        n_atom_types,
        n_formal_charge,
        n_imp_H,
        n_edge_features,
    ]
    values = {
        "atom_types": atom_types,
        "formal_charge": formal_charge,
        "imp_H": imp_H,
        "chirality": ["None", "R", "S"],
        "n_atom_types": n_atom_types,
        "n_formal_charge": n_formal_charge,
        "n_imp_H": n_imp_H,
        "n_chirality": 0,
        "n_node_features": n_node_features,
        "n_edge_features": n_edge_features,
        "bondtype_to_int": bondtype_to_int,
        "int_to_bondtype": {v: k for k, v in bondtype_to_int.items()},
        "max_n_nodes": MAX_N_NODES,
        "use_aromatic_bonds": False,
        "use_canon": True,
        "use_chirality": False,
        "use_explicit_H": False,
        "ignore_H": False,
        "decoding_route": "bfs",
        "dim_f_add": dim_f_add,
        "dim_f_conn": [MAX_N_NODES, n_edge_features],
        "len_f_add_per_node": int(np.prod(dim_f_add[1:])),
        "len_f_conn_per_node": n_edge_features,
        "device": "cpu",
        "big_positive": 1e6,
        "big_negative": -1e6,
        "condition_dim": condition_dim,
        "condition_embedding_dim": 8,
        "condition_type": "virtual_node",
        "hidden_node_features": 12,
        "enn_depth": 2,
        "enn_hidden_dim": 12,
        "enn_dropout_p": 0.0,
        "mlp1_depth": 2,
        "mlp1_hidden_dim": 12,
        "mlp1_dropout_p": 0.0,
        "mlp2_depth": 2,
        "mlp2_hidden_dim": 12,
        "mlp2_dropout_p": 0.0,
        "gather_att_depth": 2,
        "gather_att_hidden_dim": 12,
        "gather_att_dropout_p": 0.0,
        "gather_emb_depth": 2,
        "gather_emb_hidden_dim": 12,
        "gather_emb_dropout_p": 0.0,
        "gather_width": 12,
        "message_passes": 2,
        "message_size": 12,
    }
    return namedtuple("CONSTANTS", values)(**values)


def _model(condition_dim: int):
    from gnn.mpnn import GGNN

    torch.manual_seed(0)
    model = GGNN(constants=_constants(condition_dim))
    model.eval()  # dropout is zero, but eval removes any doubt about the diff
    return model


def _but_1_en_3_yne_batch(constants):
    """
    C#C-C=C, padded to max_n_nodes.

    All three bond types appear, so every per-bond-type message MLP is on the
    path and the gradient check below means something.
    """
    nodes = torch.zeros(1, MAX_N_NODES, constants.n_node_features)
    n_h = (1, 0, 1, 2)
    for i, h in enumerate(n_h):
        nodes[0, i, 0] = 1.0  # carbon
        nodes[0, i, constants.n_atom_types] = 1.0  # formal charge 0
        nodes[0, i, constants.n_atom_types + 1 + h] = 1.0

    edges = torch.zeros(1, MAX_N_NODES, MAX_N_NODES, constants.n_edge_features)
    for i, j, bond in ((0, 1, 2), (1, 2, 0), (2, 3, 1)):
        edges[0, i, j, bond] = 1.0
        edges[0, j, i, bond] = 1.0
    return nodes, edges


def test_output_shape_is_unchanged_by_conditioning():
    """Conditioning does not change the length of the action probabilities.

    Necessary but not sufficient for correct stripping: dropping the wrong node
    keeps this shape. `test_virtual_node_is_stripped_before_readout` is what
    pins down which node was removed."""
    plain = _model(condition_dim=0)
    conditional = _model(condition_dim=2)
    nodes, edges = _but_1_en_3_yne_batch(plain.constants)

    with torch.no_grad():
        unconditional_out = plain(nodes, edges)
        conditional_out = conditional(nodes, edges, torch.zeros(1, 2))

    assert unconditional_out.shape == conditional_out.shape


def test_virtual_node_is_stripped_before_readout():
    """
    The readout must see the real nodes, not the seed.

    Checking the output length is not enough: removing the last row instead of
    the seed row keeps the shape and leaves the readout reading the seed's
    hidden state as if it were node 0, with every real node shifted by one.
    This leaves the first slot empty of any real atom, so the hidden state
    there must stay zero -- a padding node has no neighbors, never receives a
    message, and is never updated. If the seed is still in place, that row
    carries the condition embedding and is not zero.
    """
    model = _model(condition_dim=2)
    constants = model.constants

    nodes = torch.zeros(1, MAX_N_NODES, constants.n_node_features)
    edges = torch.zeros(1, MAX_N_NODES, MAX_N_NODES, constants.n_edge_features)
    # Slot 0 left as padding; two bonded carbons in slots 1 and 2.
    for i in (1, 2):
        nodes[0, i, 0] = 1.0
        nodes[0, i, constants.n_atom_types] = 1.0
        nodes[0, i, constants.n_atom_types + 1 + 1] = 1.0
    edges[0, 1, 2, 0] = 1.0
    edges[0, 2, 1, 0] = 1.0

    seen = {}
    real_readout = model.readout

    def capture(hidden_nodes, nodes_in, node_mask, **kwargs):
        seen["hidden_nodes"] = hidden_nodes.detach().clone()
        seen["nodes_in"] = nodes_in.detach().clone()
        return real_readout(hidden_nodes, nodes_in, node_mask, **kwargs)

    model.readout = capture
    try:
        with torch.no_grad():
            model(nodes, edges, torch.tensor([[1.5, -0.5]]))
    finally:
        del model.readout

    hidden_nodes, nodes_in = seen["hidden_nodes"], seen["nodes_in"]
    assert hidden_nodes.shape[1] == MAX_N_NODES
    assert nodes_in.shape[1] == MAX_N_NODES, "nodes_in still carries the seed row"
    padding = nodes_in.sum(-1) == 0
    assert padding[0, 0], "slot 0 was meant to be padding"
    assert torch.count_nonzero(hidden_nodes[padding]) == 0, (
        "the readout received a non-zero hidden state for a padding slot, so the "
        "virtual seed node was not the row that got stripped"
    )


def test_condition_vector_changes_the_logits():
    model = _model(condition_dim=2)
    nodes, edges = _but_1_en_3_yne_batch(model.constants)

    with torch.no_grad():
        low = model(nodes, edges, torch.tensor([[-2.0, 0.5]]))
        high = model(nodes, edges, torch.tensor([[3.0, -1.5]]))

    assert not torch.allclose(low, high), "the condition does not reach the output"


def test_condition_reaches_the_empty_graph():
    """
    The first action of every molecule is taken on an empty graph. No real node
    exists there to carry the seed's message, so the condition can only arrive
    through the embedding concatenated to the graph embedding.
    """
    model = _model(condition_dim=2)
    nodes = torch.zeros(1, MAX_N_NODES, model.constants.n_node_features)
    edges = torch.zeros(1, MAX_N_NODES, MAX_N_NODES, model.constants.n_edge_features)

    with torch.no_grad():
        low = model(nodes, edges, torch.tensor([[-2.0, 0.5]]))
        high = model(nodes, edges, torch.tensor([[3.0, -1.5]]))

    assert not torch.allclose(low, high)


def test_every_parameter_receives_gradient():
    model = _model(condition_dim=2)
    model.train()
    nodes, edges = _but_1_en_3_yne_batch(model.constants)

    model(nodes, edges, torch.tensor([[1.0, -1.0]])).sum().backward()

    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is None or torch.all(parameter.grad == 0)
    ]
    assert not missing, f"no gradient reached: {missing}"


def test_virtual_edge_mlp_is_unused_without_conditioning():
    """
    ``msg_nns[n_edge_features]`` serves the virtual seed edge. An unconditional
    model builds it and never uses it, which is why the gradient check above
    runs only on the conditional model.
    """
    model = _model(condition_dim=0)
    model.train()
    nodes, edges = _but_1_en_3_yne_batch(model.constants)

    model(nodes, edges).sum().backward()

    virtual = model.msg_nns[model.constants.n_edge_features]
    assert all(p.grad is None or torch.all(p.grad == 0) for p in virtual.parameters())
    assert model.condition_encoder is None


class TestConditionalWarmStart:
    """
    Loading an unconditional checkpoint into a conditional model.

    All three conditional experiments resume from their program's unconditional
    prior, which the architecture-key guard used to refuse outright, so no
    conditional model could be trained as configured. A conditional model is a
    strict superset of the unconditional one, so the load is possible; what
    makes it a *warm start* rather than a partial load is that the condition's
    two pathways into the model start at zero, so the model begins computing
    exactly what the prior computed. Without that, the conditional experiment
    would be measuring a different initialization as well as conditioning.
    """

    def _warm_started(self, condition_dim=3):
        import util

        prior = _model(condition_dim=0)
        conditional = _model(condition_dim=condition_dim)
        util._warm_start_conditional(conditional, prior.state_dict())
        conditional.eval()
        return prior, conditional

    def test_the_shared_weights_are_the_prior_s(self):
        prior, conditional = self._warm_started()
        prior_state = prior.state_dict()
        shared = 0
        for name, tensor in conditional.state_dict().items():
            if name not in prior_state or name.startswith("msg_nns.3."):
                continue
            source = prior_state[name]
            if source.shape == tensor.shape:
                assert torch.equal(tensor, source), name
                shared += 1
        assert shared > 10

    def test_the_new_condition_columns_are_zero_not_random(self):
        _, conditional = self._warm_started()
        width = conditional.condition_embedding_dim
        assert width > 0
        for net in ("fAddNet2", "fConnNet2", "fTermNet2"):
            weight = getattr(conditional.ActionProbReadout, net).seq[0].weight
            assert torch.all(weight[:, -width:] == 0.0), net
            # The pretrained columns are not zero, so this is a warm start and
            # not a reinitialized layer.
            assert torch.any(weight[:, :-width] != 0.0), net

    def test_the_untrained_virtual_edge_net_is_zeroed(self):
        """
        The virtual-edge message net exists in an unconditional checkpoint,
        carrying its initialization because nothing ever trained it. Loading
        those weights would inject noise into every real node from step one.
        """
        _, conditional = self._warm_started()
        virtual = conditional.msg_nns[conditional.constants.n_edge_features]
        output = virtual.seq[len(virtual.seq) - 1]
        assert torch.all(output.weight == 0.0)
        assert torch.all(output.bias == 0.0)

    def test_the_warm_started_model_computes_what_the_prior_computed(self):
        """
        The whole point. Message aggregation is a sum and the seed is stripped
        before readout, so a zeroed condition pathway contributes nothing and
        the logits must match the prior's to numerical tolerance -- for ANY
        condition vector, including one the model will later learn from.
        """
        prior, conditional = self._warm_started()
        nodes, edges = _but_1_en_3_yne_batch(prior.constants)

        with torch.no_grad():
            expected = prior(nodes, edges)
            for condition in (
                torch.zeros(1, 3),
                torch.tensor([[1.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 1.0]]),
                torch.tensor([[5.0, -3.0, 2.0]]),
            ):
                actual = conditional(nodes, edges, condition_vector=condition)
                assert torch.allclose(actual, expected, atol=1e-5), condition

    def test_the_condition_still_learns(self):
        """
        Starting at zero must not mean staying at zero.

        The zeroed columns do receive gradient on the very first backward pass,
        because the gradient of a weight is its input times the upstream
        gradient and the condition embedding is not zero. The *encoder* does
        not: the gradient reaching it is the zeroed weight transposed times the
        upstream gradient, which is zero on both of its pathways. So the
        encoder is dead for exactly one step and alive from the second, once
        the readout columns have moved off zero -- the same mechanism a
        zero-initialized adapter relies on. Both halves are asserted, because
        "dead for one step" and "dead forever" differ by one optimizer step and
        nothing else would distinguish them.
        """
        _, conditional = self._warm_started()
        conditional.train()
        nodes, edges = _but_1_en_3_yne_batch(conditional.constants)
        condition = torch.tensor([[1.0, 0.0, 0.0]])
        width = conditional.condition_embedding_dim

        conditional(nodes, edges, condition_vector=condition).sum().backward()

        # Step one: the readout's condition columns move, the encoder does not.
        for net in ("fAddNet2", "fConnNet2", "fTermNet2"):
            grad = getattr(conditional.ActionProbReadout, net).seq[0].weight.grad
            assert torch.any(grad[:, -width:] != 0.0), net
        assert all(
            torch.all(p.grad == 0.0) for p in conditional.condition_encoder.parameters()
        )

        optimizer = torch.optim.SGD(conditional.parameters(), lr=0.1)
        optimizer.step()
        optimizer.zero_grad()

        # Step two: the pathway is open, so the encoder learns.
        conditional(nodes, edges, condition_vector=condition).sum().backward()
        for name, parameter in conditional.condition_encoder.named_parameters():
            assert parameter.grad is not None, name
            assert torch.any(parameter.grad != 0.0), f"condition_encoder.{name}"

    def test_the_model_stops_matching_the_prior_once_it_has_trained(self):
        """
        The equality above is an initialization property, not a permanent one.
        If it survived training, the conditioning would be frozen.
        """
        prior, conditional = self._warm_started()
        nodes, edges = _but_1_en_3_yne_batch(prior.constants)
        condition = torch.tensor([[1.0, 0.0, 0.0]])

        conditional.train()
        optimizer = torch.optim.SGD(conditional.parameters(), lr=0.1)
        for _ in range(3):
            optimizer.zero_grad()
            conditional(nodes, edges, condition_vector=condition).sum().backward()
            optimizer.step()

        conditional.eval()
        with torch.no_grad():
            assert not torch.allclose(
                conditional(nodes, edges, condition_vector=condition),
                prior(nodes, edges),
                atol=1e-5,
            )

    def test_a_mismatch_that_is_not_the_conditioning_still_fails(self):
        import util

        prior = _model(condition_dim=0)
        state = prior.state_dict()
        name = "gru.weight_ih"
        state[name] = state[name][:, :-1].clone()
        with pytest.raises(ValueError) as raised:
            util._warm_start_conditional(_model(condition_dim=3), state)
        assert name in str(raised.value)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_condition_vector_dtype_and_batch(batch_size):
    model = _model(condition_dim=2)
    nodes, edges = _but_1_en_3_yne_batch(model.constants)
    nodes = nodes.repeat(batch_size, 1, 1)
    edges = edges.repeat(batch_size, 1, 1, 1)
    conditions = torch.randn(batch_size, 2, dtype=torch.float32)

    with torch.no_grad():
        out = model(nodes, edges, conditions)

    assert out.shape[0] == batch_size
    assert out.dtype == torch.float32


def test_condition_dropout_zeroes_a_tenth_of_the_batch_and_is_seeded():
    """
    The unconditional baseline is only a real measurement when the model has
    been trained on the zero vector, so a dropout that silently does nothing
    turns the conditional-minus-baseline gap into an artifact. Nothing about
    that failure is visible: training converges, the metrics are all present,
    and the gap looks like a result.
    """
    import sys as _sys
    import types

    import torch

    # Workflow imports Analyzer, which imports `metrics` at module scope, so
    # src/ has to be on the path as well as src/graphinvent.
    _src = str(Path(__file__).resolve().parent.parent / "src")
    if _src not in _sys.path:
        _sys.path.insert(0, _src)
    from Workflow import Workflow

    workflow = Workflow.__new__(Workflow)
    workflow.constants = types.SimpleNamespace(condition_dropout=0.1, seed=1)
    workflow._condition_rng = None

    conditions = torch.ones(10000, 3)
    dropped = workflow._drop_conditions(conditions)
    zeroed = int((dropped.sum(dim=1) == 0).sum())
    assert 0.08 < zeroed / 10000 < 0.12

    # A surviving row is untouched, not merely non-zero.
    survivors = dropped[dropped.sum(dim=1) > 0]
    assert torch.equal(survivors, torch.ones_like(survivors))

    # Seeded: the same job drops the same examples on a rerun.
    again = Workflow.__new__(Workflow)
    again.constants = workflow.constants
    again._condition_rng = None
    assert torch.equal(dropped, again._drop_conditions(conditions))

    # And a rate of zero is a no-op rather than a near-no-op.
    off = Workflow.__new__(Workflow)
    off.constants = types.SimpleNamespace(condition_dropout=0.0, seed=1)
    off._condition_rng = None
    assert torch.equal(conditions, off._drop_conditions(conditions))
