import time
import logging

import torch

from data.noise import noise_torch

logger = logging.getLogger(__name__)


def train_freq_SR(
    args,
    fr_module,
    fr_optimizer,
    fr_criterion,
    fr_scheduler,
    train_loader,
    val_loader,
    epoch,
    tb_writer,
):
    """
    Modified from: https://github.com/sreyas-mohan/DeepFreq
    Train the frequency-representation module for one epoch
    """
    epoch_start_time = time.time()
    fr_module.train()
    loss_train_fr = 0
    for clean_signal, target_fr, _ in train_loader:
        clean_signal, target_fr = clean_signal.to(args.device), target_fr.to(args.device)
        noisy_signal = noise_torch(clean_signal, args=args)
        fr_optimizer.zero_grad()
        output_fr = fr_module(noisy_signal)
        loss_fr = fr_criterion(output_fr, target_fr)
        loss_fr.backward()
        fr_optimizer.step()
        loss_train_fr += loss_fr.data.item()

    fr_module.eval()
    loss_val_fr = 0
    for noisy_signal, _, target_fr, _ in val_loader:
        noisy_signal, target_fr = noisy_signal.to(args.device), target_fr.to(args.device)
        with torch.no_grad():
            output_fr = fr_module(noisy_signal)
        loss_fr = fr_criterion(output_fr, target_fr)
        loss_val_fr += loss_fr.data.item()

    loss_train_fr /= args.n_training
    loss_val_fr /= args.n_validation

    tb_writer.add_scalar(f"fr_{args.loss_fn}_training", loss_train_fr, epoch)
    tb_writer.add_scalar(f"fr_{args.loss_fn}_validation", loss_val_fr, epoch)

    fr_scheduler.step(loss_val_fr)
    logger.info(
        "Epochs: %d / %d, Time: %.1f, FR training loss %.2f, FR validation loss %.2f",
        epoch,
        args.n_epochs_fr,
        time.time() - epoch_start_time,
        loss_train_fr,
        loss_val_fr,
    )
