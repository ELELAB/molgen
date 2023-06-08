import os

from molgen.utils import get_root_directory
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem
from rdkit.Chem import AllChem

root_dir = get_root_directory()
os.chdir(root_dir)
config = OmegaConf.load(os.path.join(root_dir, "config", "config.yaml"))


def main(config: DictConfig) -> None:
    """Summary line.

    Script to test out different attributes of small molecules. Check calculation times and how easy it is to calculate.

    Args:
        config (DictConfig): Configuration file located in config/config.yaml

    Returns:
        None
    """
    # Defining some test smiles
    smiles = [
        "CC(C)c1cccc(c1)c2cccc3c2NC(=O)N(C3)C",
        "CC(C)c1cccc(c1)c2ccc3c(c2)c(c[nH]3)C[NH3+]",
        "CC(C)c1cccc(c1)c2ccc(=O)n(c2)CC(F)(F)F",
        "CC(C)c1cccc(c1)c2cccc3c2C(C(=O)NC3)(C)C",
        "CCCOc1ccc(cc1)c2cc(ccn2)CSC",
        "CCCOc1ccc(cc1)c2c(nccn2)O[C@@H](C)CC",
    ]

    # Converting smiles to RDKit molecules
    molecules = [Chem.MolFromSmiles(i) for i in smiles]

    # Calculating the number of heavy atoms
    [i.GetNumHeavyAtoms() for i in molecules]

    # Calculate shape properties
    molecule_volume = []
    for mol in molecules:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())

        volume = AllChem.ComputeMolVolume(mol)  # molecular volume

        molecule_volume.append(volume)

    # Calculate molecular weight
    [Chem.rdMolDescriptors.CalcExactMolWt(i) for i in molecules]


if __name__ == "__main__":  # pragma: no cover
    main(config)
