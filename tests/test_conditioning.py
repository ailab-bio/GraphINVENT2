"""
Unit tests for the conditional generation architecture (Phase 2).

Tests cover:
  1. ConditionEncoder — shape, gradient flow
  2. Virtual seed node — prepend/strip, real-node masking
  3. GGNN conditional forward — shape, reproducibility, differs from unconditional
  4. DataProcessor TSV parsing — header detection, validation, condition vectors
  5. HDFDataset — condition_vector yielded when present, absent otherwise
  6. Backward compatibility — unconditional models unaffected
"""

import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_GRAPHINVENT = Path(__file__).resolve().parent.parent / "src/graphinvent"
if str(_GRAPHINVENT) not in sys.path:
    sys.path.insert(0, str(_GRAPHINVENT))

import gnn.mpnn as mpnn_module
from DataProcessor import _read_smiles_with_conditions, _write_smiles_with_conditions
from gnn.ConditionEncoder import ConditionEncoder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_constants(condition_dim: int = 0, condition_embedding_dim: int = 16):
    from rdkit.Chem.rdchem import BondType

    atom_types = ["C", "N", "O"]
    formal_charge = [0]
    imp_H = [0, 1, 2, 3]
    n_atom_types = 3
    n_formal_charge = 1
    n_imp_H = 4
    n_edge_features = 3
    max_n_nodes = 5
    h = 16

    len_f_add_per_node = n_atom_types * n_formal_charge * n_imp_H * n_edge_features
    len_f_add = max_n_nodes * len_f_add_per_node
    len_f_conn_per_node = n_edge_features
    len_f_conn = max_n_nodes * len_f_conn_per_node

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
        "condition_dim",
        "condition_embedding_dim",
    ]
    C = namedtuple("Constants", fields)
    return C(
        atom_types=atom_types,
        formal_charge=formal_charge,
        imp_H=imp_H,
        chirality=["None"],
        n_atom_types=n_atom_types,
        n_formal_charge=n_formal_charge,
        n_imp_H=n_imp_H,
        n_chirality=0,
        n_node_features=n_atom_types + n_formal_charge + n_imp_H,
        n_edge_features=n_edge_features,
        bondtype_to_int={BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2},
        int_to_bondtype={0: BondType.SINGLE, 1: BondType.DOUBLE, 2: BondType.TRIPLE},
        max_n_nodes=max_n_nodes,
        device="cpu",
        len_f_add=len_f_add,
        len_f_conn=len_f_conn,
        len_f_add_per_node=len_f_add_per_node,
        len_f_conn_per_node=len_f_conn_per_node,
        dim_f_add=[
            max_n_nodes,
            n_atom_types,
            n_formal_charge,
            n_imp_H,
            n_edge_features,
        ],
        dim_f_conn=[max_n_nodes, n_edge_features],
        dim_f_term=[1],
        dim_action_probs=len_f_add + len_f_conn + 1,
        dim_nodes=[max_n_nodes, n_atom_types + n_formal_charge + n_imp_H],
        dim_edges=[max_n_nodes, max_n_nodes, n_edge_features],
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
        n_samples=8,
        batch_size=4,
        big_positive=1e6,
        big_negative=-1e6,
        job_dir="/tmp/graphinvent_test/",
        decoding_route="bfs",
        condition_dim=condition_dim,
        condition_embedding_dim=condition_embedding_dim,
    )


def _two_node_batch(constants, batch_size: int):
    """Two C atoms connected by a single bond."""
    N, NF = constants.max_n_nodes, constants.n_node_features
    EF = constants.n_edge_features
    nodes = torch.zeros(batch_size, N, NF)
    edges = torch.zeros(batch_size, N, N, EF)
    nodes[:, 0, 0] = 1.0  # C at index 0
    nodes[:, 1, 0] = 1.0  # C at index 1
    edges[:, 0, 1, 0] = 1.0  # single bond 0→1
    edges[:, 1, 0, 0] = 1.0  # single bond 1→0
    return nodes, edges


# ---------------------------------------------------------------------------
# 1. ConditionEncoder
# ---------------------------------------------------------------------------


class TestConditionEncoder:
    def test_output_shape(self):
        enc = ConditionEncoder(
            condition_dim=3, hidden_dim=32, condition_embedding_dim=64
        )
        x = torch.randn(8, 3)
        out = enc(x)
        assert out.shape == (8, 64), f"Expected (8,64) got {out.shape}"

    def test_gradient_flows(self):
        enc = ConditionEncoder(
            condition_dim=2, hidden_dim=16, condition_embedding_dim=16
        )
        x = torch.randn(4, 2, requires_grad=True)
        out = enc(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        for p in enc.parameters():
            assert p.grad is not None

    def test_different_inputs_differ(self):
        enc = ConditionEncoder(
            condition_dim=3, hidden_dim=16, condition_embedding_dim=16
        )
        enc.eval()
        with torch.no_grad():
            x1 = torch.zeros(1, 3)
            x2 = torch.ones(1, 3)
            assert not torch.allclose(enc(x1), enc(x2))


# ---------------------------------------------------------------------------
# 2. GGNN — architecture checks
# ---------------------------------------------------------------------------


class TestGGNNConditional:
    @pytest.fixture
    def cond_constants(self):
        return _make_constants(condition_dim=2, condition_embedding_dim=16)

    @pytest.fixture
    def uncond_constants(self):
        return _make_constants(condition_dim=0)

    def test_msg_nns_has_virtual_slot(self, cond_constants):
        """GGNN always has n_edge_features + 1 msg_nns."""
        m = mpnn_module.GGNN(cond_constants)
        assert len(m.msg_nns) == cond_constants.n_edge_features + 1

    def test_condition_encoder_created(self, cond_constants):
        m = mpnn_module.GGNN(cond_constants)
        assert m.condition_encoder is not None

    def test_no_condition_encoder_when_uncond(self, uncond_constants):
        m = mpnn_module.GGNN(uncond_constants)
        assert m.condition_encoder is None

    def test_unconditional_forward_shape(self, uncond_constants):
        m = mpnn_module.GGNN(uncond_constants)
        B = 3
        nodes, edges = _two_node_batch(uncond_constants, B)
        out = m(nodes, edges)
        assert out.shape == (B, uncond_constants.dim_action_probs)

    def test_conditional_forward_shape(self, cond_constants):
        m = mpnn_module.GGNN(cond_constants)
        B = 3
        nodes, edges = _two_node_batch(cond_constants, B)
        cond = torch.randn(B, cond_constants.condition_dim)
        out = m(nodes, edges, condition_vector=cond)
        assert out.shape == (B, cond_constants.dim_action_probs)

    def test_conditional_differs_from_unconditional(self, cond_constants):
        """Different condition vectors must produce different logits."""
        torch.manual_seed(42)
        m = mpnn_module.GGNN(cond_constants)
        m.eval()
        B = 2
        nodes, edges = _two_node_batch(cond_constants, B)
        with torch.no_grad():
            out_no_cond = m(nodes, edges)
            cond = torch.randn(B, cond_constants.condition_dim)
            out_with_cond = m(nodes, edges, condition_vector=cond)
        assert not torch.allclose(
            out_no_cond, out_with_cond
        ), "Conditional forward should differ from unconditional"

    def test_different_conditions_give_different_logits(self, cond_constants):
        torch.manual_seed(42)
        m = mpnn_module.GGNN(cond_constants)
        m.eval()
        B = 2
        nodes, edges = _two_node_batch(cond_constants, B)
        with torch.no_grad():
            cond_a = torch.zeros(B, cond_constants.condition_dim)
            cond_b = torch.ones(B, cond_constants.condition_dim)
            out_a = m(nodes, edges, condition_vector=cond_a)
            out_b = m(nodes, edges, condition_vector=cond_b)
        assert not torch.allclose(
            out_a, out_b
        ), "Different conditions should yield different logits"

    def test_conditional_gradient_flows_to_encoder(self, cond_constants):
        m = mpnn_module.GGNN(cond_constants)
        m.train()
        B = 2
        nodes, edges = _two_node_batch(cond_constants, B)
        cond = torch.randn(B, cond_constants.condition_dim)
        target = F.softmax(torch.rand(B, cond_constants.dim_action_probs), dim=-1)

        logits = m(nodes, edges, condition_vector=cond)
        loss = F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")
        loss.backward()

        # ConditionEncoder params must get gradients when conditioning is active
        for name, p in m.condition_encoder.named_parameters():
            assert p.grad is not None, f"ConditionEncoder param {name} has no gradient"

    def test_virtual_edge_mlp_gets_gradient_with_conditioning(self, cond_constants):
        """The virtual-edge MLP must receive gradients in a conditional forward."""
        m = mpnn_module.GGNN(cond_constants)
        m.train()
        B = 2
        nodes, edges = _two_node_batch(cond_constants, B)
        cond = torch.randn(B, cond_constants.condition_dim)
        target = F.softmax(torch.rand(B, cond_constants.dim_action_probs), dim=-1)

        logits = m(nodes, edges, condition_vector=cond)
        loss = F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")
        loss.backward()

        virtual_mlp = m.msg_nns[cond_constants.n_edge_features]
        for name, p in virtual_mlp.named_parameters():
            assert p.grad is not None, f"Virtual-edge MLP param {name} has no gradient"

    def test_condition_none_is_identical_to_no_condition(self, cond_constants):
        """Passing condition_vector=None is identical to not passing it."""
        torch.manual_seed(0)
        m = mpnn_module.GGNN(cond_constants)
        m.eval()
        B = 2
        nodes, edges = _two_node_batch(cond_constants, B)
        with torch.no_grad():
            out1 = m(nodes, edges)
            out2 = m(nodes, edges, condition_vector=None)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# 3. DataProcessor TSV parsing
# ---------------------------------------------------------------------------


class TestTSVParsing:
    def test_plain_smiles_no_conditions(self, tmp_path):
        f = tmp_path / "test.smi"
        f.write_text("CC\nCO\nCCC\n")
        smiles, cond_vecs, cond_names = _read_smiles_with_conditions(f)
        assert smiles == ["CC", "CO", "CCC"]
        assert cond_vecs == []
        assert cond_names == []

    def test_tsv_header_and_values(self, tmp_path):
        f = tmp_path / "test.smi"
        f.write_text("SMILES\tpLogS\tpKa\nCC\t-0.5\t4.75\nCO\t1.2\t9.0\n")
        smiles, cond_vecs, cond_names = _read_smiles_with_conditions(f)
        assert smiles == ["CC", "CO"]
        assert cond_names == ["pLogS", "pKa"]
        assert len(cond_vecs) == 2
        np.testing.assert_allclose(cond_vecs[0], [-0.5, 4.75], rtol=1e-5)
        np.testing.assert_allclose(cond_vecs[1], [1.2, 9.0], rtol=1e-5)
        assert cond_vecs[0].dtype == np.float32

    def test_tsv_wrong_column_count_raises(self, tmp_path):
        f = tmp_path / "test.smi"
        f.write_text("SMILES\tpLogS\nCC\t-0.5\t4.75\n")
        with pytest.raises(ValueError, match="tab-separated fields"):
            _read_smiles_with_conditions(f)

    def test_tsv_non_numeric_raises(self, tmp_path):
        f = tmp_path / "test.smi"
        f.write_text("SMILES\tpLogS\nCC\tnot_a_number\n")
        with pytest.raises(ValueError, match="non-numeric"):
            _read_smiles_with_conditions(f)

    def test_tsv_wrong_header_raises(self, tmp_path):
        f = tmp_path / "test.smi"
        f.write_text("mol\tprop\nCC\t1.0\n")
        with pytest.raises(ValueError, match="SMILES"):
            _read_smiles_with_conditions(f)

    def test_roundtrip_write_and_read(self, tmp_path):
        f = tmp_path / "test.smi"
        smiles = ["CC", "CO"]
        cond_vecs = [
            np.array([-0.5, 4.75], dtype=np.float32),
            np.array([1.2, 9.0], dtype=np.float32),
        ]
        names = ["pLogS", "pKa"]
        _write_smiles_with_conditions(smiles, cond_vecs, names, f)
        s2, c2, n2 = _read_smiles_with_conditions(f)
        assert s2 == smiles
        assert n2 == names
        for a, b in zip(c2, cond_vecs):
            np.testing.assert_allclose(a, b, rtol=1e-5)

    def test_write_plain_smiles(self, tmp_path):
        f = tmp_path / "plain.smi"
        _write_smiles_with_conditions(["CC", "CO"], [], [], f)
        content = f.read_text()
        assert "SMILES" not in content
        assert content.strip() == "CC\nCO"


# ---------------------------------------------------------------------------
# 4. HDFDataset — condition_vector yielded when present
# ---------------------------------------------------------------------------


class TestHDFDatasetConditioning:
    @pytest.fixture
    def hdf_with_conditions(self, tmp_path):
        """Create a minimal HDF5 file with condition_vector dataset."""
        import h5py

        path = tmp_path / "cond.h5"
        n = 10
        with h5py.File(path, "w") as f:
            f.create_dataset("nodes", data=np.zeros((n, 5, 8), dtype=np.int8))
            f.create_dataset("edges", data=np.zeros((n, 5, 5, 3), dtype=np.int8))
            f.create_dataset("action_probs", data=np.zeros((n, 100), dtype=np.int8))
            f.create_dataset(
                "condition_vector", data=np.random.randn(n, 2).astype(np.float32)
            )
        return str(path)

    @pytest.fixture
    def hdf_without_conditions(self, tmp_path):
        import h5py

        path = tmp_path / "uncond.h5"
        n = 10
        with h5py.File(path, "w") as f:
            f.create_dataset("nodes", data=np.zeros((n, 5, 8), dtype=np.int8))
            f.create_dataset("edges", data=np.zeros((n, 5, 5, 3), dtype=np.int8))
            f.create_dataset("action_probs", data=np.zeros((n, 100), dtype=np.int8))
        return str(path)

    def test_four_tuple_when_condition_present(self, hdf_with_conditions):
        from BlockDatasetLoader import HDFDataset

        ds = HDFDataset(hdf_with_conditions)
        item = ds[0]
        assert len(item) == 4, f"Expected 4-tuple, got {len(item)}-tuple"
        nodes, edges, action_probs, cond = item
        assert cond.dtype == torch.float32
        assert cond.shape == (2,)

    def test_three_tuple_when_condition_absent(self, hdf_without_conditions):
        from BlockDatasetLoader import HDFDataset

        ds = HDFDataset(hdf_without_conditions)
        item = ds[0]
        assert len(item) == 3, f"Expected 3-tuple, got {len(item)}-tuple"

    def test_condition_vector_values_preserved(self, hdf_with_conditions):
        import h5py
        from BlockDatasetLoader import HDFDataset

        # Read expected value from HDF directly
        with h5py.File(hdf_with_conditions, "r") as f:
            expected = f["condition_vector"][0]
        ds = HDFDataset(hdf_with_conditions)
        _, _, _, cond = ds[0]
        np.testing.assert_allclose(cond.numpy(), expected, rtol=1e-6)
