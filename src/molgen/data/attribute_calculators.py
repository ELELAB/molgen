from jazzy.api import deltag_from_smiles, molecular_vector_from_smiles
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors

from src.molgen.sascorer.sascorer import calculateScore


class RdkitAttributeCalculator:
    def __init__(self):
        pass

    def calculate_all(self, smile):
        try:
            molecule = Chem.MolFromSmiles(smile)
            qed = Chem.QED.qed(molecule)
            logp = Crippen.MolLogP(molecule)
            tpsa = Chem.rdMolDescriptors.CalcTPSA(molecule)
            weight = Descriptors.ExactMolWt(molecule)

        except Exception as e:
            print("Error in calculating qed, tpsa, logp or weight attributes for smile: ", smile)
            print("Error: ", e)
            qed = None
            logp = None
            tpsa = None
            weight = None
            volume = None
            return [qed, logp, tpsa, weight, volume]

        try:
            molecule = Chem.AddHs(molecule)
            AllChem.EmbedMolecule(molecule)
            volume = AllChem.ComputeMolVolume(molecule)

        except Exception as e:
            print("Error in calculating volume for smile: ", smile)
            print("Error: ", e)
            volume = None

        return [qed, logp, tpsa, weight, volume]

    def add_all_to_file(
        self,
        dataframe,
        smile_column="smiles",
        qed_column="qed",
        logp_column="logp",
        tpsa_column="tpsa",
        weight_column="weight",
        volume_column="volume",
    ):
        (
            dataframe[qed_column],
            dataframe[logp_column],
            dataframe[tpsa_column],
            dataframe[weight_column],
            dataframe[volume_column],
        ) = zip(*dataframe[smile_column].apply(lambda x: self.calculate_all(x)))
        return dataframe


class JazzyAttributeCalculator:
    def __init__(self):
        pass

    def calculate_molecular_vector(self, smile):
        try:
            mol_vector = molecular_vector_from_smiles(smile)
            sdc = mol_vector["sdc"]
            sdx = mol_vector["sdx"]
            sa = mol_vector["sa"]
            dga = mol_vector["dga"]
            dgp = mol_vector["dgp"]
            dgtot = mol_vector["dgtot"]
            mds = sdc + sdx
        except Exception as e:
            print("Error in calculating molecular vector for smile: ", smile)
            print("Error: ", e)
            sdc = None
            sdx = None
            sa = None
            dga = None
            dgp = None
            dgtot = None
            mds = None
        return [sdc, sdx, sa, dga, dgp, dgtot, mds]

    def calculate_deltag(self, smile):
        try:
            deltag = deltag_from_smiles(smile)
        except Exception as e:
            print("Error in calculating deltag for smile: ", smile)
            print("Error: ", e)
            deltag = None
        return deltag

    def add_molecular_vector_to_file(
        self,
        dataframe,
        smile_column="smiles",
        sdc_column="sdc",
        sdx_column="sdx",
        sa_column="sa",
        dga_column="dga",
        dgp_column="dgp",
        dgtot_column="dgtot",
        mds_column="mds",
    ):
        (
            dataframe[sdc_column],
            dataframe[sdx_column],
            dataframe[sa_column],
            dataframe[dga_column],
            dataframe[dgp_column],
            dataframe[dgtot_column],
            dataframe[mds_column],
        ) = zip(*dataframe[smile_column].apply(lambda x: self.calculate_molecular_vector(x)))
        return dataframe

    def add_deltag_to_file(self, dataframe, smile_column="smiles", deltag_column="deltag"):
        dataframe[deltag_column] = dataframe[smile_column].apply(lambda x: self.calculate_deltag(x))
        return dataframe


class SAScorerAttributeCalculator:
    def __init__(self):
        pass

    def calculate_sascorer(self, smile):
        try:
            molecule = Chem.MolFromSmiles(smile)
            sascore = calculateScore(molecule)
        except Exception as e:
            print("Error in calculating sascorer for smile: ", smile)
            print("Error: ", e)
            sascore = None
        return sascore

    def add_sascorer_to_file(self, dataframe, smile_column="smiles", sascorer_column="sascore"):
        dataframe[sascorer_column] = dataframe[smile_column].apply(lambda x: self.calculate_sascorer(x))
        return dataframe
