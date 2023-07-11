import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
import numpy as np
import pandas as pd
from src.molgen.data.attribute_calculators import (
    JazzyAttributeCalculator,
    RdkitAttributeCalculator,
    SAScorerAttributeCalculator,
)
import torch

def calculate_validity(generated_smiles):
    # Checking validity of smiles
    valid_smiles = []
    for smile in generated_smiles:
        if Chem.MolFromSmiles(smile) != None:
            valid_smiles.append(smile)
    
    validity = len(valid_smiles) / len(generated_smiles)

    return validity

def calculate_uniqueness(generated_smiles):

    fingerprints = set()
    unique_molecules = []
    
    for molecule in generated_smiles:
        mol = Chem.MolFromSmiles(molecule)
        if mol is not None:
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)  # Circular fingerprint
            fingerprint_str = fingerprint.ToBitString()
            if fingerprint_str not in fingerprints:
                fingerprints.add(fingerprint_str)
                unique_molecules.append(molecule)
    
    uniqueness_score = len(unique_molecules) / len(generated_smiles)

    return uniqueness_score

def calculate_similarity(generated_smiles, training_smiles):
    gen_similarities = []
    for molecule in generated_smiles:
        generated_mol = Chem.MolFromSmiles(molecule)
        if generated_mol is None:
            continue
        
        generated_fp = AllChem.GetMorganFingerprintAsBitVect(generated_mol, 2, nBits=1024)  # Circular fingerprint

        similarities = []
        for training_smile in training_smiles:
            training_mol = Chem.MolFromSmiles(training_smile)
            if training_mol is not None:
                training_fp = AllChem.GetMorganFingerprintAsBitVect(training_mol, 2, nBits=1024)  # Circular fingerprint
                similarity = TanimotoSimilarity(generated_fp, training_fp)
                similarities.append(similarity)
        
        if similarities:
            max_similarity = max(similarities)
            gen_similarities.append(max_similarity)

    return np.mean(gen_similarities) 

def calculate_attribute_accuracy(generated_smiles, target_conditions, config, scaler=None):

    # Defining the attribute calculators
    rdkit_attribute_calculator = RdkitAttributeCalculator()
    jazzy_attribute_calculator = JazzyAttributeCalculator()
    sascorer_attribute_calculator = SAScorerAttributeCalculator()

    # Make DataFrame of target generated_smiles and conditions
    generated_smiles_df = pd.DataFrame({"smiles": generated_smiles})

    # Calculating the attributes of the generated molecules and adding them to the dataframe
    if "volume" in config.data.attribute_columns:
        generated_smiles_df = rdkit_attribute_calculator.add_all_to_file(
            generated_smiles_df, smile_column="smiles"
        )  # Takes 0.004 seconds per molecule for qed, logp, tpsa, weight, but 0.25 for volume
    else: 
        generated_smiles_df = rdkit_attribute_calculator.add_non_volume_to_file(
            generated_smiles_df, smile_column="smiles"
        )

    # Add the sascorer attributes to the file
    if "sascore" in config.data.attribute_columns:
        generated_smiles_df = sascorer_attribute_calculator.add_sascorer_to_file(
            generated_smiles_df, smile_column="smiles"
        )

    # Add the jazzy attributes to the file
    generated_smiles_df = jazzy_attribute_calculator.add_molecular_vector_to_file(
        generated_smiles_df, smile_column="smiles"
    )  # Takes around 0.1 seconds per molecule

    # Getting relevant conditions
    generated_smiles_conditions = generated_smiles_df[config.data.attribute_columns].values
    non_nan_idx = np.sum(generated_smiles_conditions == None, axis=1) == 0

    # Removing nan values
    generated_smiles_conditions = generated_smiles_conditions[non_nan_idx]
    target_conditions = target_conditions[non_nan_idx]

    if sum(non_nan_idx) == 0:
        return 1000000
    # Scaling the conditions
    if scaler is not None:
        # Calculating the accuracy
        generated_smiles_conditions = torch.Tensor(scaler.transform(generated_smiles_conditions))
        attribute_accuracy = torch.mean(torch.abs(generated_smiles_conditions - target_conditions))
    else: 
        # Calculating the accuracy
        attribute_accuracy = torch.mean(torch.abs(generated_smiles_conditions - target_conditions) / torch.max(target_conditions, axis=0)[0])

    return attribute_accuracy

def calculate_novelty(generated_molecules, reference_molecules):
    unique_generated = set(generated_molecules)
    unique_reference = set(reference_molecules)
    
    novel_molecules = unique_generated - unique_reference
    novelty = len(novel_molecules) / len(unique_generated)
    
    return novelty
