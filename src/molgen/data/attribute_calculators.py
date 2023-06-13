import os
import sys
import time

import sascorer
from jazzy.api import deltag_from_smiles, molecular_vector_from_smiles
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, RDConfig


class RdkitAttributeCalculator:
    def __init__(self):
        pass

    # def add_qed_to_file(self, dataframe, smile_column="smiles", qed_column="qed"):
    #     dataframe[qed_column] = dataframe[smile_column].apply(lambda x: self.calculate_qed(x))
    #     return dataframe

    # def add_logp_to_file(self, dataframe, smile_column="smiles", logp_column="logp"):
    #     dataframe[logp_column] = dataframe[smile_column].apply(lambda x: self.calculate_logp(x))
    #     return dataframe

    # def add_tpsa_to_file(self, dataframe, smile_column="smiles", tpsa_column="tpsa"):
    #     dataframe[tpsa_column] = dataframe[smile_column].apply(lambda x: self.calculate_tpsa(x))
    #     return dataframe

    # def add_weight_to_file(self, dataframe, smile_column="smiles", weight_column="weight"):
    #     dataframe[weight_column] = dataframe[smile_column].apply(lambda x: self.calculate_weight(x))
    #     return dataframe

    # def add_volume_to_file(self, dataframe, smile_column="smiles", volume_column="volume"):
    #     dataframe[volume_column] = dataframe[smile_column].apply(lambda x: self.calculate_volume(x))
    #     return dataframe

    # def calculate_qed(self, smile):
    #     molecule = Chem.MolFromSmiles(smile)
    #     qed = Chem.QED.qed(molecule)
    #     return qed

    # def calculate_logp(self, smile):
    #     molecule = Chem.MolFromSmiles(smile)
    #     logp = Crippen.MolLogP(molecule)
    #     return logp

    # def calculate_tpsa(self, smile):
    #     molecule = Chem.MolFromSmiles(smile)
    #     tpsa = Chem.rdMolDescriptors.CalcTPSA(molecule)
    #     return tpsa

    # def calculate_weight(self, smile):
    #     molecule = Chem.MolFromSmiles(smile)
    #     weight = Descriptors.ExactMolWt(molecule)
    #     return weight

    # def calculate_volume(self, smile):
    #     molecule = Chem.AddHs(Chem.MolFromSmiles(smile))
    #     AllChem.EmbedMolecule(molecule)
    #     volume = AllChem.ComputeMolVolume(molecule)
    #     return volume

    def calculate_all(self, smile):
        molecule = Chem.MolFromSmiles(smile)
        qed = Chem.QED.qed(molecule)
        logp = Crippen.MolLogP(molecule)
        tpsa = Chem.rdMolDescriptors.CalcTPSA(molecule)
        weight = Descriptors.ExactMolWt(molecule)
        time.time()
        molecule = Chem.AddHs(molecule)
        AllChem.EmbedMolecule(molecule)
        volume = AllChem.ComputeMolVolume(molecule)
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
        mol_vector = molecular_vector_from_smiles(smile)
        sdc = mol_vector["sdc"]
        sdx = mol_vector["sdx"]
        sa = mol_vector["sa"]
        dga = mol_vector["dga"]
        dgp = mol_vector["dgp"]
        dgtot = mol_vector["dgtot"]
        mds = sdc + sdx
        return [sdc, sdx, sa, dga, dgp, dgtot, mds]

    def calculate_deltag(self, smile):
        deltag = deltag_from_smiles(smile)
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
        # To be able to import the sascorer from the rdkit contrib directory we need to add it to the path temporarily
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))

    def calculate_sascorer(self, smile):
        start = time.time()
        molecule = Chem.MolFromSmiles(smile)
        sascore = sascorer.calculateScore(molecule)
        print("Time taken to calculate sascorer: ", time.time() - start)
        return sascore

    def add_sascorer_to_file(self, dataframe, smile_column="smiles", sascorer_column="sascore"):
        dataframe[sascorer_column] = dataframe[smile_column].apply(lambda x: self.calculate_sascorer(x))
        return dataframe
