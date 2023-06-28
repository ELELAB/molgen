import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.DataStructs import FingerprintSimilarity


class LossFunction:
    def __init__(self, padding_int, n_attributes=0):
        self.padding_int = padding_int
        self.n_attributes = n_attributes

    def pretrain_loss(self, smile_prediction_p, smile_target, beta, gamma, mu, log_var):
        """
        Loss used in the GCT paper. This is a combination of cross_entropy and kl_divergence.
        loss = cross_entropy + beta * kl_divergence + gamma * orthogonal_loss
            - kl_divergence is the part which makes sure that encoder outputs of mean/variance follows a ~Norm(0,1) distribution
            - cross_entropy is the part that makes the model output smiles.
        """

        # Making smile prediction vector 2 dimensional. (merges all elements from all smiles in batch).
        smile_prediction_p = smile_prediction_p.contiguous().view(-1, smile_prediction_p.shape[-1])
        smile_target = smile_target.view(-1)

        # Calculating cross entropy loss (Reconstruction loss)
        smile_rce_loss = F.cross_entropy(smile_prediction_p, smile_target, reduction="mean")

        # Calculating kl_divergence, this will be 0 if mean=0 and std=1
        kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())  # Is 0 when mu = 0, std = 1

        # Calculating orthogonal loss
        latent_std = torch.exp(0.5 * log_var)
        cov_matrix = torch.matmul(latent_std.unsqueeze(-1), latent_std.unsqueeze(-2))
        gram_matrix = torch.matmul(mu.unsqueeze(-1), mu.unsqueeze(-2))
        orthogonal_loss = torch.norm(cov_matrix * gram_matrix, dim=(-1, -2)).mean()

        # Final loss where kl_divergence is weighted by beta, to make sure it does not take over too soon.
        loss = smile_rce_loss + beta * kl_divergence + gamma * orthogonal_loss

        return loss, smile_rce_loss, kl_divergence, orthogonal_loss

    def finetune_loss(
        self,
        smile_prediction_p,
        smile_target,
        beta,
        gamma,
        mu,
        log_var,
        smile_attributes,
        targeted_attributes,
        pretrain_weight,
        attribute_weight,
    ):
        """
        Loss to finetune the model, to generate molecules with specific attributes.
        loss = pretrain_weight * pretrain_loss + finetune_weight * attribute_loss
        """

        # Getting the pretrained loss and weighting it
        pretrain_loss = self.pretrain_loss(smile_prediction_p, smile_target, beta, gamma, mu, log_var)

        # Getting the finetune loss
        smile_attributes = smile_attributes.view(-1, smile_attributes.shape[-1])
        targeted_attributes = targeted_attributes.view(-1, targeted_attributes.shape[-1])
        attribute_loss = F.mse_loss(smile_attributes, targeted_attributes, reduction="mean")

        # Final loss
        loss = pretrain_weight * pretrain_loss + attribute_weight * attribute_loss

        return loss, pretrain_loss, attribute_loss

    def generation_loss(
        self, generated_molecules, molecule_attributes, targeted_attributes, attribute_weight, diversity_weight
    ):
        """
        Loss of generated molecules. This loss makes sure that the generated molecules have the targeted attributes,
        and that the molecules generated are different from each other.
        """

        # Getting the attribute loss
        molecule_attributes = molecule_attributes.view(-1, molecule_attributes.shape[-1])
        targeted_attributes = targeted_attributes.view(-1, targeted_attributes.shape[-1])
        attribute_loss = F.mse_loss(molecule_attributes, targeted_attributes, reduction="mean")

        # Getting the diversity loss using the tanimoto similarity
        generated_molecules = generated_molecules.view(-1)
        similarities = []
        fingerprints = [Chem.RDKFingerprint(mol) for mol in generated_molecules]
        for i in range(len(generated_molecules)):
            for j in range(i + 1, len(generated_molecules)):
                similarity = FingerprintSimilarity(fingerprints[i], fingerprints[j])
                similarities.append(similarity)

        diversity_loss = torch.mean(torch.tensor(similarities))

        # Final loss
        loss = attribute_weight * attribute_loss + diversity_weight * diversity_loss

        return loss, attribute_loss, diversity_loss
