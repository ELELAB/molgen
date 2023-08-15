import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as MorganFingerprint
from rdkit.DataStructs import BulkTanimotoSimilarity, TanimotoSimilarity
from tqdm import tqdm

from src.molgen.data.attribute_calculators import (
    JazzyAttributeCalculator,
    RdkitAttributeCalculator,
    SAScorerAttributeCalculator,
)


def calculate_validity(generated_smiles):
    # Checking validity of smiles
    valid_smiles = []
    for smile in generated_smiles:
        if Chem.MolFromSmiles(smile) is not None:
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
    # Getting device
    device = target_conditions.device

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
    try:
        non_nan_idx = np.sum(np.isnan(generated_smiles_conditions), axis=1) == 0
    except TypeError:
        non_nan_idx = np.sum(generated_smiles_conditions is None, axis=1) == 0

    # Adding target values to df
    target_columns = [i + "_target" for i in config.data.attribute_columns]
    if scaler is not None:
        if type(target_conditions) == torch.Tensor:
            target_conditions_backscaled = scaler.inverse_transform(target_conditions.cpu().numpy())
        else:
            target_conditions_backscaled = scaler.inverse_transform(target_conditions)
        generated_smiles_df[target_columns] = target_conditions_backscaled
    else:
        generated_smiles_df[target_columns] = target_conditions

    # Removing nan values
    generated_smiles_conditions = generated_smiles_conditions[non_nan_idx]
    target_conditions = target_conditions[non_nan_idx]

    if sum(non_nan_idx) == 0:
        return 1000000, 0, generated_smiles_conditions
    # Scaling the conditions
    if scaler is not None:
        # Calculating the accuracy
        generated_smiles_conditions = torch.Tensor(scaler.transform(generated_smiles_conditions)).to(device)
        attribute_accuracy = torch.mean(torch.abs(generated_smiles_conditions - target_conditions))
    else:
        # Calculating the accuracy
        generated_smiles_conditions = torch.Tensor(generated_smiles_conditions).to(device)
        attribute_accuracy = torch.mean(
            torch.abs(generated_smiles_conditions - target_conditions) / torch.max(target_conditions, axis=0)[0]
        )

    # Calculate validity, where it was possible to calculate the attributes
    validity = sum(non_nan_idx) / len(non_nan_idx)

    return attribute_accuracy.item(), validity, generated_smiles_df


def calculate_novelty(generated_molecules, reference_molecules):
    unique_generated = set(generated_molecules)
    unique_reference = set(reference_molecules)

    novel_molecules = unique_generated - unique_reference
    novelty = len(novel_molecules) / len(unique_generated)

    return novelty


def compute_fingerprint_similarity(fp1, fp2):
    return BulkTanimotoSimilarity(fp1, fp2)


def calculate_fragment_similarity(generated_set, reference_set):
    print("Calculating fragment similarity")
    gen_fps = []
    for mol in tqdm(generated_set, desc="Generating fingerprints for generated molecules"):
        try:
            gen_fps.append(AllChem.GetMorganFingerprint(mol, 2, nBits=1024))
        except Exception:
            continue
    ref_fps = []
    for mol in tqdm(reference_set, desc="Generating fingerprints for reference molecules"):
        try:
            ref_fps.append(AllChem.GetMorganFingerprint(mol, 2, nBits=1024))
        except Exception:
            continue

    intersection = sum((gen_fp & ref_fp).CountOnBits() for gen_fp, ref_fp in zip(gen_fps, ref_fps))
    norm_gen_fps = sum(fp.CountOnBits() for fp in gen_fps)
    norm_ref_fps = sum(fp.CountOnBits() for fp in ref_fps)

    fragment_similarity = intersection / (norm_gen_fps * norm_ref_fps) ** 0.5
    return fragment_similarity


def calculate_scaffold_similarity(generated_set, reference_set):
    gen_scaffolds = []
    for mol in tqdm(generated_set, desc="Generating scaffolds for generated molecules"):
        try:
            gen_scaffolds.append(Chem.MolToMurckoScaffold(mol, includeChirality=True))
        except Exception:
            continue
    ref_scaffolds = []
    for mol in tqdm(reference_set, desc="Generating scaffolds for reference molecules"):
        try:
            ref_scaffolds.append(Chem.MolToMurckoScaffold(mol, includeChirality=True))
        except Exception:
            continue

    gen_scaffolds = set(gen_scaffolds)
    ref_scaffolds = set(ref_scaffolds)

    intersection = sum(1 for scaffold in gen_scaffolds if scaffold in ref_scaffolds)
    norm_gen_scaffolds = len(gen_scaffolds)
    norm_ref_scaffolds = len(ref_scaffolds)

    scaffold_similarity = intersection / (norm_gen_scaffolds * norm_ref_scaffolds) ** 0.5
    return scaffold_similarity


def calculate_snn_similarity(generated_set, reference_set):
    gen_fps = [AllChem.GetMorganFingerprint(mol, 2, nBits=1024) for mol in generated_set]
    ref_fps = [AllChem.GetMorganFingerprint(mol, 2, nBits=1024) for mol in reference_set]

    snn_similarity = sum(
        max(compute_fingerprint_similarity(gen_fp, ref_fp)) for gen_fp in gen_fps for ref_fp in ref_fps
    ) / len(generated_set)
    return snn_similarity


def calculate_internal_diversity(generated_set, p):
    gen_fps = [AllChem.GetMorganFingerprint(mol, 2, nBits=1024) for mol in generated_set]
    int_diversity = 1 - (
        sum(sum(compute_fingerprint_similarity(fp1, fp2) ** p for fp1 in gen_fps) for fp2 in gen_fps)
        / (len(generated_set) ** 2)
    )
    return int_diversity


# Example usage:
# generated_set = [Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('CCN'), Chem.MolFromSmiles('CCOC')]
# reference_set = [Chem.MolFromSmiles('CCCC'), Chem.MolFromSmiles('CCOC')]
# frag_similarity = calculate_fragment_similarity(generated_set, reference_set)
# scaff_similarity = calculate_scaffold_similarity(generated_set, reference_set)
# snn_similarity = calculate_snn_similarity(generated_set, reference_set)
# int_diversity = calculate_internal_diversity(generated_set, 2)
# print(frag_similarity, scaff_similarity, snn_similarity, int_diversity


def calculate_cosine_similarity(fragment_counts_G, fragment_counts_R):
    numerator = sum(fragment_counts_G[fragment] * fragment_counts_R[fragment] for fragment in fragment_counts_G)
    denominator_G = math.sqrt(sum(count**2 for count in fragment_counts_G.values()))
    denominator_R = math.sqrt(sum(count**2 for count in fragment_counts_R.values()))
    similarity = numerator / (denominator_G * denominator_R)
    return similarity


def calculate_fragment_counts(smiles_list):
    fragment_counts = defaultdict(int)

    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fgs = AllChem.FragmentOnBRICSBonds(mol)
            fragments = Chem.MolToSmiles(fgs).split(".")
            for fragment in fragments:
                fragment_counts[fragment] += 1

    return fragment_counts


def make_fingerprints(smiles):
    fps = []
    smiles = np.unique(np.array(smiles, dtype="str"))
    for smile in tqdm(smiles):
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            fp = np.asarray(MorganFingerprint(mol, 2, nBits=1024), dtype="uint8")
            fps.append(fp)

    return np.vstack(fps)


def average_agg_tanimoto(stock_vecs, gen_vecs, batch_size=5000, agg="max", device="cpu", p=1):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ["max", "mean"], "Can aggregate only max or mean"  # noqa: S101
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j : j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i : i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) + y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == "max":
                agg_tanimoto[i : i + y_gen.shape[1]] = np.maximum(agg_tanimoto[i : i + y_gen.shape[1]], jac.max(0))
            elif agg == "mean":
                agg_tanimoto[i : i + y_gen.shape[1]] += jac.sum(0)
                total[i : i + y_gen.shape[1]] += jac.shape[0]
    if agg == "mean":
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto) ** (1 / p)
    return np.mean(agg_tanimoto)


def calculate_intdiv_snn_frag(generated_set, reference_set, all_ps, max_time_pr_metric=60 * 60 * 12):
    print("Making molecule fragment counts")
    gen_fragment_counts = calculate_fragment_counts(generated_set)
    ref_fragment_counts = calculate_fragment_counts(reference_set)

    print("Calculating fragment similarity")
    frag_similarity = calculate_cosine_similarity(gen_fragment_counts, ref_fragment_counts)

    print("Generating fingerprints for generated and reference molecules")
    gen_fps = make_fingerprints(generated_set)
    ref_fps = make_fingerprints(reference_set)

    print("Calculating snn similarity")
    snn_similarity = average_agg_tanimoto(gen_fps, ref_fps, agg="max", device="cuda")

    p_div = []
    for p in all_ps:
        int_diversity = 1 - (average_agg_tanimoto(gen_fps, gen_fps, agg="mean", device="cuda", p=p)).mean()
        p_div.append(int_diversity)

    return frag_similarity, snn_similarity, p_div


# def calculate_intdiv_snn_frag(generated_set, reference_set, all_ps, max_time_pr_metric=60*60*12):
#     print(f"Generating fingerprints for generated and reference molecules")
#     gen_fps = []
#     for smile in tqdm(generated_set, desc="Generating fingerprints for generated molecules"):
#         mol = Chem.MolFromSmiles(smile)
#         try:
#             gen_fps.append(AllChem.GetMorganFingerprint(mol, 2))
#         except:
#             continue
#     ref_fps = []
#     for smile in tqdm(reference_set, desc="Generating fingerprints for reference molecules"):
#         mol = Chem.MolFromSmiles(smile)
#         try:
#             ref_fps.append(AllChem.GetMorganFingerprint(mol, 2))
#         except:
#             continue

#     print(f"Calculating fragment similarity")
#     intersection = sum(TanimotoSimilarity(gen_fp, ref_fp) for gen_fp, ref_fp in zip(gen_fps, ref_fps))
#     norm_gen_fps = len(gen_fps)
#     norm_ref_fps = len(ref_fps)
#     frag_similarity = intersection / (norm_gen_fps + norm_ref_fps - intersection)

#     print(f"Calculating snn similarity")
#     max_similarities = []
#     t1_snn = time.time()
#     for gen_fp in tqdm(gen_fps, desc="Calculating max similarities"):
#         max_sim = 0
#         for ref_fp in ref_fps:
#             sim = TanimotoSimilarity(gen_fp, ref_fp)
#             if sim > max_sim:
#                 max_sim = sim
#         max_similarities.append(max_sim)
#         if time.time() - t1_snn > max_time_pr_metric:
#             print(f"Time limit reached for snn similarity")
#             break

#     snn_similarity = sum(max_similarities) / len(max_similarities)

#     print(f"Calculating internal diversity")
#     p_div = []
#     for p in all_ps:
#         t1_p = time.time()
#         count = 0
#         sumsum = 0
#         for fp1 in tqdm(gen_fps, desc=f"Calculating internal diversity with p={p}"):
#             for fp2 in gen_fps:
#                 sumsum += TanimotoSimilarity(fp1, fp2) ** p
#                 count += 1
#             if time.time() - t1_p > max_time_pr_metric:
#                 print(f"Time limit reached for internal diversity with p={p}")
#                 break
#         int_diversity = 1 - (sumsum / (count))
#         p_div.append(int_diversity)

#     return frag_similarity, snn_similarity, p_div


# from scipy.spatial.distance import cosine as cos_distance

# def calculate_cosine_similarity(ref_counts, gen_counts):
#     """
#     Computes cosine similarity between
#      dictionaries of form {name: count}. Non-present
#      elements are considered zero:

#      sim = <r, g> / ||r|| / ||g||
#     """
#     if len(ref_counts) == 0 or len(gen_counts) == 0:
#         return np.nan
#     keys = np.unique(list(ref_counts.keys()) + list(gen_counts.keys()))
#     ref_vec = np.array([ref_counts.get(k, 0) for k in keys])
#     gen_vec = np.array([gen_counts.get(k, 0) for k in keys])
#     return 1 - cos_distance(ref_vec, gen_vec)
