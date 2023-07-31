import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import wandb
from src.molgen.models.utils import make_nopeak_mask, make_padding_mask
from src.molgen.utils import get_root_directory


def pretrain_model(
    config,
    model,
    device,
    trainloader,
    valloader,
    klannealer,
    orthannealer,
    attributeannealer,
    optimizer,
    scheduler,
    loss_function,
    n_conditions,
    n_epochs,
    symbol_to_index,
    wandb_run,
    highest_epoch=0,
    save_model=False,
    save_model_epoch=10,
    save_name="pretrained",
):
    """
    Function to train the model.
    """
    # Getting the root directory
    root_dir = get_root_directory()

    # Getting number of batches
    num_train_batches = len(trainloader)
    num_val_batches = len(valloader)
    steps = max((highest_epoch - 1) * num_train_batches, 0)
    val_steps = max((highest_epoch - 1) * num_val_batches, 0)
    log_n_steps = config.wandb.log_n_steps

    # Training the model
    for epoch in tqdm(range(n_epochs - highest_epoch)):
        # Calculating the beta and gamma for the epoch for weighting the loss function
        beta = klannealer.calculate_beta(steps)
        gamma = orthannealer.calculate_beta(steps)
        theta = attributeannealer.calculate_beta(steps)

        # Training the model
        model.train()
        train_loss = 0
        train_rce_loss = 0
        train_kl_divergence = 0
        train_orthogonal_loss = 0
        train_attribute_loss = 0
        train_accuracy = 0

        # Clearing the optimizer gradient
        optimizer.zero_grad()

        # Looping over the batches
        for _idx, (src, trg_input, trg_output, attributes) in enumerate(trainloader):
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
            selfie_p, _, _, mu, logvar, attributes_pred = model(src, trg_input, attributes, src_mask, trg_mask)

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.hist(mu.detach().cpu().numpy().flatten(), bins=100, log=True)
            # plt.xlim(-1, 1)
            # plt.savefig(root_dir + "/figures/mu.png")

            # std = torch.exp(0.5 * logvar)

            # plt.figure()
            # plt.hist(std.detach().cpu().numpy().flatten(), bins=100, log=True)
            # plt.xlim(0, 2)
            # plt.savefig(root_dir + "/figures/std.png")

            # Calculating the loss
            loss, rce_loss, kl_divergence, orthogonal_loss, attribute_loss = loss_function.pretrain_loss(
                selfie_p, trg_output, beta, gamma, theta, mu, logvar, attributes, attributes_pred
            )

            # Perform backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Clear the gradients
            optimizer.zero_grad()

            # Adding the loss to the total loss
            accuracy = torch.argmax(selfie_p, axis=2) == trg_output
            accuracy = torch.mean(accuracy.type(torch.float32))
            train_accuracy += accuracy.item() / num_train_batches
            train_loss += loss.item()
            train_rce_loss += rce_loss.item()
            train_kl_divergence += kl_divergence.item()
            train_orthogonal_loss += orthogonal_loss.item()
            train_attribute_loss += attribute_loss.item()

            # Logging the loss
            if steps % log_n_steps == 0:
                # Logging the loss
                loss_contributions = np.vstack(
                    (
                        rce_loss.item(),
                        kl_divergence.item() * beta,
                        orthogonal_loss.item() * gamma,
                        attribute_loss.item() * theta,
                    )
                ).T
                loss_contributions_names = ["rce_loss", "kl_divergence", "orthogonal_loss", "attribute_loss"]

                plt.figure(figsize=(10, 6))
                plt.bar(loss_contributions_names, np.mean(loss_contributions, axis=0))
                plt.xlabel("Loss Names")
                plt.ylabel("Mean Loss Value")
                plt.title("Contributions of Different Losses to the Final Loss")
                plt.grid(axis="y")

                wandb_img = wandb.Image(plt)

                wandb_run.log(
                    {
                        "train_loss": loss.item(),
                        "train_rce_loss": rce_loss.item(),
                        "train_kl_divergence": kl_divergence.item(),
                        "train_orthogonal_loss": orthogonal_loss.item(),
                        "train_attribute_loss": attribute_loss.item(),
                        "train_accuracy": accuracy.item(),
                        "train_beta": beta,
                        "train_gamma": gamma,
                        "train_theta": theta,
                        "train_steps": steps,
                        "train_loss_contributions": wandb_img,
                    }
                )
                print(f"Epoch: {epoch+highest_epoch}, Batch: {_idx} / {num_train_batches}, Loss: {loss.item()}")

            steps += 1

            # Update beta and gamma
            if steps % config.gct.kla_increment_steps == 0 or steps % config.gct.orth_increment_steps == 0:
                beta = klannealer.calculate_beta(steps)
                gamma = orthannealer.calculate_beta(steps)
                theta = attributeannealer.calculate_beta(steps)
                if save_model and epoch % save_model_epoch == 0 and not config.gct.debugging:
                    torch.save(
                        model.state_dict(),
                        os.path.join(root_dir, config.gct.model_save_path, save_name + f"_{epoch}.pt"),
                    )

        # Logging the average loss for the epoch
        wandb_run.log(
            {
                "train_loss": train_loss / num_train_batches,
                "train_rce_loss": train_rce_loss / num_train_batches,
                "train_kl_divergence": train_kl_divergence / num_train_batches,
                "train_orthogonal_loss": train_orthogonal_loss / num_train_batches,
                "train_attribute_loss": train_attribute_loss / num_train_batches,
                "train_accuracy": train_accuracy,
                "train_beta": beta,
                "train_gamma": gamma,
                "train_epoch": epoch + highest_epoch,
            }
        )

        # Validating the model
        model.eval()
        val_accuracy = 0
        val_loss = 0
        val_rce_loss = 0
        val_kl_divergence = 0
        val_orthogonal_loss = 0
        val_attribute_loss = 0
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
                selfie_p, _, _, mu, logvar, attributes_pred = model(src, trg_input, attributes, src_mask, trg_mask)

                # Calculating the loss
                loss, rce_loss, kl_divergence, orthogonal_loss, attribute_loss = loss_function.pretrain_loss(
                    selfie_p, trg_output, beta, gamma, theta, mu, logvar, attributes, attributes_pred
                )

                # Adding the loss to the total loss
                accuracy = torch.argmax(selfie_p, axis=2) == trg_output
                accuracy = torch.mean(accuracy.type(torch.float32))
                val_steps += 1
                val_accuracy += accuracy.item() / num_val_batches
                val_loss += loss.item()
                val_rce_loss += rce_loss.item()
                val_kl_divergence += kl_divergence.item()
                val_orthogonal_loss += orthogonal_loss.item()
                val_attribute_loss += attribute_loss.item()

        # Updating the learning rate scheduler
        scheduler.step(val_loss)

        # Logging the average loss for the epoch
        wandb_run.log(
            {
                "val_loss": val_loss / num_val_batches,
                "val_rce_loss": val_rce_loss / num_val_batches,
                "val_kl_divergence": val_kl_divergence / num_val_batches,
                "val_orthogonal_loss": val_orthogonal_loss / num_val_batches,
                "val_attribute_loss": val_attribute_loss / num_val_batches,
                "val_accuracy": val_accuracy,
                "val_beta": beta,
                "val_gamma": gamma,
                "val_epoch": epoch + highest_epoch,
            }
        )

        # Saving the model
        if save_model and epoch % save_model_epoch == 0 and not config.gct.debugging:
            torch.save(
                model.state_dict(), os.path.join(root_dir, config.gct.model_save_path, save_name + f"_{epoch}.pt")
            )

    return (
        model,
        val_loss / num_val_batches,
        val_rce_loss / num_val_batches,
        val_kl_divergence / num_val_batches,
        val_orthogonal_loss / num_val_batches,
        val_attribute_loss / num_val_batches,
        val_accuracy,
    )


def pretrain_model_ctd(
    config,
    model,
    device,
    trainloader,
    valloader,
    optimizer,
    scheduler,
    loss_function,
    n_conditions,
    n_epochs,
    symbol_to_index,
    wandb_run,
    highest_epoch=0,
    save_model=False,
    save_name="pretrained",
):
    """
    Function to train the model.
    """
    # Getting the root directory
    root_dir = get_root_directory()

    # Getting number of batches
    num_train_batches = len(trainloader)
    num_val_batches = len(valloader)
    steps = max((highest_epoch - 1) * num_train_batches, 0)
    val_steps = max((highest_epoch - 1) * num_val_batches, 0)
    log_n_steps = config.wandb.log_n_steps

    # Training the model
    for epoch in tqdm(range(n_epochs - highest_epoch)):
        # Training the model
        model.train()
        train_loss = 0
        train_accuracy = 0

        # Clearing the optimizer gradient
        optimizer.zero_grad()

        # Looping over the batches
        for _idx, (_, trg_input, trg_output, attributes) in enumerate(trainloader):
            # If debugging and steps > 500, break the loop to save time
            if config.ctd.debugging and steps > 500:
                break

            # Getting batch size
            batch_size = trg_input.shape[0]

            # Move data to device
            trg_output = trg_output.to(device)
            trg_input = trg_input.to(device)
            attributes = attributes.to(device)

            # Defining the masks
            src_mask = None
            trg_no_peak_mask = make_nopeak_mask(
                batch_size, device=device, dimension=trg_output.shape[1], n_conditions=0
            )
            trg_padding_mask = make_padding_mask(trg_input, symbol_to_index["[nop]"], n_conditions=0)

            trg_mask = torch.logical_and(trg_no_peak_mask, trg_padding_mask)

            # Making forward pass
            selfie_p = model(trg_input, attributes, src_mask, trg_mask)

            # Calculating the loss
            loss = loss_function.pretrain_loss(selfie_p, trg_output)

            # Perform backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Clear the gradients
            optimizer.zero_grad()

            # Adding the loss to the total loss
            accuracy = torch.argmax(selfie_p, axis=2) == trg_output
            accuracy = torch.mean(accuracy.type(torch.float32))
            train_accuracy += accuracy.item() / num_train_batches
            train_loss += loss.item()

            # Logging the loss
            if steps % log_n_steps == 0:
                wandb_run.log(
                    {
                        "train_loss": loss.item(),
                        "train_accuracy": accuracy.item(),
                        "train_steps": steps,
                    }
                )
                print(f"Epoch: {epoch+highest_epoch}, Batch: {_idx} / {num_train_batches}, Loss: {loss.item()}")

            steps += 1

            # Update beta and gamma
            if steps % config.ctd.save_model_steps == 0:
                torch.save(
                    model.state_dict(), os.path.join(root_dir, config.ctd.model_save_path, save_name + f"_{epoch}.pt")
                )

        # Logging the average loss for the epoch
        wandb_run.log(
            {
                "train_loss": train_loss / num_train_batches,
                "train_accuracy": train_accuracy,
                "train_epoch": epoch + highest_epoch,
            }
        )

        # Validating the model
        model.eval()
        val_accuracy = 0
        val_loss = 0
        with torch.no_grad():
            for _idx, (_, trg_input, trg_output, attributes) in tqdm(
                enumerate(valloader), leave=False, total=num_val_batches
            ):
                # If debugging and steps > 500, break the loop to save time
                if config.ctd.debugging and val_steps > 50:
                    break

                # Getting batch size
                batch_size = trg_input.shape[0]

                # Move data to device
                trg_output = trg_output.to(device)
                trg_input = trg_input.to(device)
                attributes = attributes.to(device)

                # Defining the masks
                src_mask = None
                trg_no_peak_mask = make_nopeak_mask(
                    batch_size, device=device, dimension=trg_output.shape[1], n_conditions=0
                )
                trg_padding_mask = make_padding_mask(trg_input, symbol_to_index["[nop]"], n_conditions=0)

                trg_mask = torch.logical_and(trg_no_peak_mask, trg_padding_mask)

                # Making forward pass
                selfie_p = model(trg_input, attributes, src_mask, trg_mask)

                # Calculating the loss
                loss = loss_function.pretrain_loss(selfie_p, trg_output)

                # Adding the loss to the total loss
                accuracy = torch.argmax(selfie_p, axis=2) == trg_output
                accuracy = torch.mean(accuracy.type(torch.float32))
                val_steps += 1
                val_accuracy += accuracy.item() / num_val_batches
                val_loss += loss.item()

        # Updating the learning rate scheduler
        scheduler.step(val_loss)

        # Logging the average loss for the epoch
        wandb_run.log(
            {
                "val_loss": val_loss / num_val_batches,
                "val_accuracy": val_accuracy,
                "val_epoch": epoch + highest_epoch,
            }
        )

        # Saving the model
        if save_model:
            torch.save(
                model.state_dict(), os.path.join(root_dir, config.ctd.model_save_path, save_name + f"_{epoch}.pt")
            )

    return model, val_loss / num_val_batches, val_accuracy
