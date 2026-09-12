"""
Round-trip tests for the molecular graph encoding.

Every training target and every generated molecule passes through
``PreprocessingGraph`` (SMILES -> node/edge tensors) and ``graph_to_mol``
(tensors -> RDKit Mol).  If that round trip is lossy the model is trained on a
corrupted target and its reported validity is measured against the wrong
molecule, so these tests assert exact recovery rather than mere parseability.

Each test corresponds to a defect that previously passed the rest of the suite
undetected, which is why the assertions are on chemical identity rather than on
shapes or types.
"""

import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType

_GRAPHINVENT = Path(__file__).resolve().parent.parent / "src/graphinvent"
if str(_GRAPHINVENT) not in sys.path:
    sys.path.insert(0, str(_GRAPHINVENT))

RDLogger.DisableLog("rdApp.*")


def _make_constants(use_chirality: bool = False, max_n_nodes: int = 30):
    """
    Minimal constants namedtuple covering the vocabulary these molecules need.

    Kept independent of ``parameters.config`` so the tests need no dataset and
    no job directory.
    """
    atom_types = ["C", "N", "O", "S", "Cl", "F"]
    formal_charge = [-1, 0, 1]
    imp_H = [0, 1, 2, 3]
    chirality = ["None", "R", "S"]

    n_atom_types = len(atom_types)
    n_formal_charge = len(formal_charge)
    n_imp_H = len(imp_H)
    n_chirality = len(chirality) if use_chirality else 0
    n_node_features = n_atom_types + n_formal_charge + n_imp_H + n_chirality

    bondtype_to_int = {
        BondType.SINGLE: 0,
        BondType.DOUBLE: 1,
        BondType.TRIPLE: 2,
        BondType.AROMATIC: 3,
    }
    n_edge_features = len(bondtype_to_int)

    dim_f_add = [max_n_nodes, n_atom_types, n_formal_charge, n_imp_H]
    if use_chirality:
        dim_f_add.append(n_chirality)
    dim_f_add.append(n_edge_features)
    dim_f_conn = [max_n_nodes, n_edge_features]

    values = {
        "atom_types": atom_types,
        "formal_charge": formal_charge,
        "imp_H": imp_H,
        "chirality": chirality,
        "n_atom_types": n_atom_types,
        "n_formal_charge": n_formal_charge,
        "n_imp_H": n_imp_H,
        "n_chirality": n_chirality,
        "n_node_features": n_node_features,
        "n_edge_features": n_edge_features,
        "bondtype_to_int": bondtype_to_int,
        "int_to_bondtype": {v: k for k, v in bondtype_to_int.items()},
        "max_n_nodes": max_n_nodes,
        "use_aromatic_bonds": True,
        "use_canon": True,
        "use_chirality": use_chirality,
        "use_explicit_H": False,
        "ignore_H": False,
        "decoding_route": "bfs",
        "dim_f_add": dim_f_add,
        "dim_f_conn": dim_f_conn,
        "device": "cpu",
    }
    return namedtuple("CONSTANTS", values)(**values)


@pytest.fixture(scope="module")
def constants():
    return _make_constants(use_chirality=False)


@pytest.fixture(scope="module")
def chiral_constants():
    return _make_constants(use_chirality=True)


def _roundtrip(constants, smiles: str) -> str:
    from MolecularGraph import PreprocessingGraph

    graph = PreprocessingGraph(constants=constants, molecule=Chem.MolFromSmiles(smiles))
    return graph.get_smiles()


# ---------------------------------------------------------------------------
# Hydrogen bookkeeping
# ---------------------------------------------------------------------------


class TestHydrogenRoundTrip:
    """
    The implicit-H count is a node feature, so decoding has to restore it.

    Writing it to the private `_TotalNumHs` property, as the decoder once did,
    has no effect: RDKit never reads that property, so the count was discarded
    and any aromatic ring whose kekulisation depends on an N-H failed to
    sanitise and was scored invalid.
    """

    @pytest.mark.parametrize(
        "smiles",
        [
            "c1cc[nH]c1",  # pyrrole
            "c1c[nH]cn1",  # imidazole
            "c1ccc2[nH]ccc2c1",  # indole
            "c1cc[nH]n1",  # pyrazole
        ],
    )
    def test_aromatic_nh_heterocycles_survive(self, constants, smiles):
        result = _roundtrip(constants, smiles)
        assert result is not None, f"{smiles} decoded to None"
        assert Chem.MolFromSmiles(result) is not None, f"{smiles} -> unparseable"
        assert Chem.CanonSmiles(result) == Chem.CanonSmiles(smiles)

    @pytest.mark.parametrize(
        "smiles",
        ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"],
    )
    def test_common_molecules_survive(self, constants, smiles):
        assert Chem.CanonSmiles(_roundtrip(constants, smiles)) == Chem.CanonSmiles(
            smiles
        )


# ---------------------------------------------------------------------------
# Stereochemistry
# ---------------------------------------------------------------------------


class TestChiralityRoundTrip:
    """
    With ``use_chirality`` the R/S state occupies a one-hot segment of every
    node feature vector, tripling the size of the f_add tensor.  If decoding
    discards it the model pays that cost to learn a target it cannot express,
    and the two enantiomers become the same molecule.
    """

    @pytest.mark.parametrize(
        "smiles",
        [
            "C[C@H](N)C(=O)O",
            "C[C@@H](N)C(=O)O",
            # hydroxyproline fragment: the VHL-ligand core of many degraders
            "O[C@@H]1C[C@H](N)CN1C(=O)C",
        ],
    )
    def test_stereocentres_are_preserved(self, chiral_constants, smiles):
        result = _roundtrip(chiral_constants, smiles)
        assert Chem.CanonSmiles(result) == Chem.CanonSmiles(smiles)

    def test_enantiomers_do_not_collapse(self, chiral_constants):
        r = _roundtrip(chiral_constants, "C[C@H](N)C(=O)O")
        s = _roundtrip(chiral_constants, "C[C@@H](N)C(=O)O")
        assert Chem.CanonSmiles(r) != Chem.CanonSmiles(s)


# ---------------------------------------------------------------------------
# Node ordering
# ---------------------------------------------------------------------------


class TestCanonicalOrdering:
    def test_equivalent_smiles_give_identical_graphs(self, constants):
        """
        ``use_canon`` exists so that the decoding route depends on the molecule
        rather than on how its SMILES happened to be written.  The traversal
        previously started from ``atom_ranking[0]`` -- the canonical *rank of
        atom 0*, used as a node *index* -- and de-duplicated its frontier with
        ``set()``, which discarded the ranking it had just computed.
        """
        from MolecularGraph import PreprocessingGraph

        variants = [
            "CC(=O)Oc1ccccc1C(=O)O",
            "OC(=O)c1ccccc1OC(C)=O",
            "c1ccc(C(=O)O)c(OC(C)=O)c1",
        ]
        graphs = [
            PreprocessingGraph(constants=constants, molecule=Chem.MolFromSmiles(v))
            for v in variants
        ]
        for i in range(1, len(graphs)):
            assert np.array_equal(
                graphs[0].node_features, graphs[i].node_features
            ), f"node features differ for variant {i}"
            assert np.array_equal(
                graphs[0].edge_features, graphs[i].edge_features
            ), f"edge features differ for variant {i}"


# ---------------------------------------------------------------------------
# Inputs the representation cannot encode
# ---------------------------------------------------------------------------


class TestDisconnectedGraphs:
    def test_multifragment_input_raises(self, constants):
        """
        A salt has no single traversal order, and the BFS used to spin forever
        on one: the frontier emptied while nodes remained unvisited, so the
        whole preprocessing job hung with no error and no output.
        """
        from MolecularGraph import PreprocessingGraph

        with pytest.raises(ValueError, match="disconnected"):
            PreprocessingGraph(
                constants=constants, molecule=Chem.MolFromSmiles("CCO.Cl")
            )


# ---------------------------------------------------------------------------
# Decoding route
# ---------------------------------------------------------------------------


class TestDecodingRoute:
    @pytest.mark.parametrize("smiles", ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"])
    def test_route_length_matches_edge_count(self, constants, smiles):
        """One action per bond, plus the first atom and the terminate step."""
        from MolecularGraph import PreprocessingGraph

        graph = PreprocessingGraph(
            constants=constants, molecule=Chem.MolFromSmiles(smiles)
        )
        assert graph.get_decoding_route_length() == int(graph.get_n_edges()) + 2

    def test_first_state_is_terminate_on_the_full_graph(self, constants):
        """
        Index 0 of the route is the finished molecule paired with the terminate
        action; every later index is a truncation of it.
        """
        from MolecularGraph import PreprocessingGraph

        graph = PreprocessingGraph(
            constants=constants, molecule=Chem.MolFromSmiles("CCO")
        )
        _, action_probs = graph.get_decoding_route_state(subgraph_idx=0)
        assert action_probs[-1] == 1
        assert action_probs[:-1].sum() == 0
