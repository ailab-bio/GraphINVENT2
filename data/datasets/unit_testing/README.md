# Fixture Molecules for GraphINVENT Unit Testing

Ten molecules covering a progression of structural complexity and key edge cases
for testing autoregressive graph-based molecular generation.

## Molecules

| # | Name          | SMILES                                                    | Why included                                      |
|---|---------------|-----------------------------------------------------------|---------------------------------------------------|
| 1 | Methane       | `C`                                                       | Minimal case: single atom, immediate stop         |
| 2 | Ethanol       | `CCO`                                                     | Simple chain with heteroatom (O)                  |
| 3 | Benzene       | `c1ccccc1`                                                | Basic ring closure + aromaticity                  |
| 4 | Aspirin       | `CC(=O)Oc1ccccc1C(=O)O`                                  | Medium drug-like complexity, branching, ester      |
| 5 | Caffeine      | `Cn1c(=O)c2c(ncn2C)n(C)c1=O`                             | Fused bicyclic rings, N heteroatoms               |
| 6 | Adamantane    | `C1C2CC3CC1CC(C2)C3`                                     | Bridged cage topology, high symmetry (Td)         |
| 7 | Biphenyl      | `c1ccc(-c2ccccc2)cc1`                                    | Linked (non-fused) ring systems                   |
| 8 | Cubane        | `C12C3C4C1C5C3C4C25`                                     | Extreme strain, multiple 4-membered rings, Oh sym |
| 9 | Sulfasalazine | `O=C(O)c1cc(/N=N/c2ccc(NS(=O)(=O)c3ccccn3)cc2)ccc1O`   | Diverse atoms (C,N,O,S), azo + sulfonamide groups |
|10 | Azulene       | `c1ccc2cccc-2cc1`                                        | Non-benzenoid aromatic, fused 5+7 ring            |

All molecules are valid and RDKit-parseable.
