"""
Draws a grid image of molecules from a SMILES file.

Usage:
    visualize path/to/file.smi
    visualize path/to/file.smi --n 50 --ncols 5 --size 250x200 --out grid.png

Called as a script:
    python graphinvent/tools/draw_molecules.py path/to/file.smi
"""
import argparse
import os
import random
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw a grid of molecules from a SMILES file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("smi",
                        help="Path to the .smi file.")
    parser.add_argument("--n",      type=int, default=25,
                        help="Max number of molecules to draw.")
    parser.add_argument("--ncols",  type=int, default=5,
                        help="Number of columns in the grid.")
    parser.add_argument("--size",   type=str, default="250x200",
                        help="Cell size as WxH pixels.")
    parser.add_argument("--out",    type=str, default=None,
                        help="Output PNG path (default: <file>_grid.png next to input).")
    parser.add_argument("--first",  action="store_true",
                        help="Take the first N molecules instead of sampling randomly.")
    return parser


def draw_grid(smi_file: str, n: int, ncols: int, cell_size: tuple,
              out_path: str, use_random: bool) -> None:
    from rdkit.Chem.Draw import MolsToGridImage
    from rdkit.Chem.rdmolfiles import SmilesMolSupplier

    def load_molecules(path):
        with open(path) as f:
            has_header = "SMILES" in f.readline()
        return SmilesMolSupplier(path, sanitize=True, nameColumn=-1, titleLine=has_header)

    supplier = load_molecules(path=smi_file)
    mols = [mol for mol in supplier if mol is not None]

    if not mols:
        print("No valid molecules found in the input file.")
        return

    if use_random:
        mols = random.sample(mols, min(n, len(mols)))
    else:
        mols = mols[:n]

    img = MolsToGridImage(
        mols,
        molsPerRow=ncols,
        subImgSize=cell_size,
        returnPNG=False,
    )
    img.save(out_path)
    print(f"* Saved grid of {len(mols)} molecules → {out_path}")


def main() -> None:
    args = _build_parser().parse_args()

    w, h = (int(x) for x in args.size.lower().split("x"))

    if args.out is None:
        base = os.path.splitext(os.path.basename(args.smi))[0]
        out_path = os.path.join(os.path.dirname(os.path.abspath(args.smi)),
                                f"{base}_grid.png")
    else:
        out_path = args.out

    draw_grid(
        smi_file=args.smi,
        n=args.n,
        ncols=args.ncols,
        cell_size=(w, h),
        out_path=out_path,
        use_random=not args.first,
    )


if __name__ == "__main__":
    main()
