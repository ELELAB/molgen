from tqdm import tqdm
import torch
import os

from src.molgen.models.utils import make_nopeak_mask, make_padding_mask
from src.molgen.utils import get_root_directory

def pretrain_model(config, model, device, trainloader, valloader, klannealer, orthannealer, optimizer, scheduler, loss_function, n_conditions, n_epochs, symbol_to_index, wandb_run, highest_epoch=0, save_model=False, save_model_epoch=10):
    """
    Function to train the model.
    """
    # Getting the root directory
    root_dir = get_root_directory()

    # Getting number of batches
    num_train_batches = len(trainloader)
    num_val_batches = len(valloader)
    steps = 0
    val_steps = 0
    log_n_steps = config.wandb.log_n_steps

    # Training the model
    for epoch in tqdm(range(n_epochs-highest_epoch)):
        # Calculating the beta and gamma for the epoch for weighting the loss function
        beta = klannealer.calculate_beta(epoch)
        gamma = orthannealer.calculate_beta(epoch)

        # Training the model
        model.train()
        train_loss = 0
        train_rce_loss = 0
        train_kl_divergence = 0
        train_orthogonal_loss = 0
        train_accuracy = 0
        for _idx, (src, trg_input, trg_output, attributes) in tqdm(
            enumerate(trainloader), leave=False, total=num_train_batches
        ):
            # If debugging and steps > 500, break the loop to save time
            if config.gct.debugging and steps > 500:
                break
            
            # Getting batch size
            batch_size = src.shape[0]

            # Move data to device
            src = src.to(device)
            trg_output = trg_output.to(device)
            trg_input = trg_input.to(device)
            attributes = attributes.to(device)

            # Clearing the optimizer gradient
            optimizer.zero_grad()

            # Defining the masks
            src_mask = None
            if model.include_conditions_decoder:
                trg_no_peak_mask = make_nopeak_mask(
                    batch_size, device=device, dimension=trg_output.shape[1], n_conditions=n_conditions
                )
                trg_padding_mask = make_padding_mask(trg_input, symbol_to_index["[nop]"], n_conditions=n_conditions)
            else:
                trg_no_peak_mask = make_nopeak_mask(
                    batch_size, device=device, dimension=trg_output.shape[1], n_conditions=0
                )
                trg_padding_mask = make_padding_mask(trg_input, symbol_to_index["[nop]"], n_conditions=0)

            trg_mask = torch.logical_and(trg_no_peak_mask, trg_padding_mask)

            # Making forward pass
            selfie_p, _, _, mu, logvar = model(src, trg_input, attributes, src_mask, trg_mask)

            # Calculating the loss
            loss, rce_loss, kl_divergence, orthogonal_loss = loss_function.pretrain_loss(
                selfie_p, trg_output, beta, gamma, mu, logvar
            )

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Adding the loss to the total loss
            accuracy = torch.argmax(selfie_p, axis=2) == trg_output
            accuracy = torch.mean(accuracy.type(torch.float))
            train_accuracy += accuracy.item() / num_train_batches
            train_loss += loss.item()
            train_rce_loss += rce_loss.item()
            train_kl_divergence += kl_divergence.item()
            train_orthogonal_loss += orthogonal_loss.item()
            steps += 1

            # Logging the loss
            if steps % log_n_steps == 0:
                wandb_run.log(
                    {
                        "train_loss": loss.item(),
                        "train_rce_loss": rce_loss.item(),
                        "train_kl_divergence": kl_divergence.item(),
                        "train_orthogonal_loss": orthogonal_loss.item(),
                        "train_accuracy": accuracy.item(),
                        "train_beta": beta,
                        "train_gamma": gamma,
                        "train_steps": steps,
                    }
                )

        # Logging the average loss for the epoch
        wandb_run.log(
            {
                "train_loss": train_loss / num_train_batches,
                "train_rce_loss": train_rce_loss / num_train_batches,
                "train_kl_divergence": train_kl_divergence / num_train_batches,
                "train_orthogonal_loss": train_orthogonal_loss / num_train_batches,
                "train_accuracy": train_accuracy,
                "train_beta": beta,
                "train_gamma": gamma,
                "train_epoch": epoch+highest_epoch,
            }
        )

        # Validating the model
        model.eval()
        val_accuracy = 0
        val_loss = 0
        val_rce_loss = 0
        val_kl_divergence = 0
        val_orthogonal_loss = 0
        with torch.no_grad():
            for _idx, (src, trg_input, trg_output, attributes) in tqdm(
                enumerate(valloader), leave=False, total=num_val_batches
            ):
                # If debugging and steps > 500, break the loop to save time
                if config.gct.debugging and val_steps > 50:
                    break

                # Getting batch size
                batch_size = src.shape[0]

                # Move data to device
                src = src.to(device)
                trg_output = trg_output.to(device)
                trg_input = trg_input.to(device)
                attributes = attributes.to(device)

                # Defining the masks
                src_mask = None
                if model.include_conditions_decoder:
                    trg_no_peak_mask = make_nopeak_mask(
                        batch_size, device=device, dimension=trg_output.shape[1], n_conditions=n_conditions
                    )
                    trg_padding_mask = make_padding_mask(trg_input, symbol_to_index["[nop]"], n_conditions=n_conditions)
                else:
                    trg_no_peak_mask = make_nopeak_mask(
                        batch_size, device=device, dimension=trg_output.shape[1], n_conditions=0
                    )
                    trg_padding_mask = make_padding_mask(trg_input, symbol_to_index["[nop]"], n_conditions=0)

                trg_mask = torch.logical_and(trg_no_peak_mask, trg_padding_mask)

                # Making forward pass
                selfie_p, _, _, mu, logvar = model(src, trg_input, attributes, src_mask, trg_mask)

                # Calculating the loss
                loss, rce_loss, kl_divergence, orthogonal_loss = loss_function.pretrain_loss(
                    selfie_p, trg_output, beta, gamma, mu, logvar
                )

                # Adding the loss to the total loss
                accuracy = torch.argmax(selfie_p, axis=2) == trg_output
                accuracy = torch.mean(accuracy.type(torch.float))
                val_steps += 1
                val_accuracy += accuracy.item() / num_val_batches
                val_loss += loss.item()
                val_rce_loss += rce_loss.item()
                val_kl_divergence += kl_divergence.item()
                val_orthogonal_loss += orthogonal_loss.item()

        # Updating the learning rate scheduler
        scheduler.step(val_loss)

        # Logging the average loss for the epoch
        wandb_run.log(
            {
                "val_loss": val_loss / num_val_batches,
                "val_rce_loss": val_rce_loss / num_val_batches,
                "val_kl_divergence": val_kl_divergence / num_val_batches,
                "val_orthogonal_loss": val_orthogonal_loss / num_val_batches,
                "val_accuracy": val_accuracy,
                "val_beta": beta,
                "val_gamma": gamma,
                "val_epoch": epoch+highest_epoch,
            }
        )

        # Saving the model
        if save_model and epoch % save_model_epoch == 0:
            torch.save(model.state_dict(), os.path.join(root_dir, config.gct.model_save_path, f"pretrained_{epoch}.pt"))

    return model, val_loss / num_val_batches, val_rce_loss, val_kl_divergence, val_orthogonal_loss, val_accuracy

