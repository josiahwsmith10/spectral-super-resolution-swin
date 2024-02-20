import torch


def test_freq_SR(
    args,
    fr_module,
    fr_criterion,
    test_loader,
):
    fr_module.eval()
    loss_test_fr = 0
    # Testing Model...
    print("Testing Model...")
    for noisy_signal, _, target_fr, freq in test_loader:
        noisy_signal, target_fr = noisy_signal.to(args.device), target_fr.to(args.device)
        with torch.no_grad():
            output_fr = fr_module(noisy_signal)
        loss_fr = fr_criterion(output_fr, target_fr)
        loss_test_fr += loss_fr.data.item()
    loss_test_fr /= args.n_testing
    return loss_test_fr


def test_basic_SR(fr_module, x: torch.Tensor, args):
    """
    Tests a single tensor of complex-data. Returns the resulting tensor on the CPU
    after being passed through the model.
    """

    # Input checking
    assert torch.is_tensor(x), "Must input a torch tensor"
    assert (
        x.dim() == 2 or x.dim() == 3
    ), "Must be 2-D complex-valued or 3-D real-valued with layered I/Q"
    if x.dim() == 2:
        assert torch.is_complex(
            x
        ), "Must be 2-D complex-valued or 3-D real-valued with layered I/Q"
        x = torch.stack((x.real, x.imag), dim=1)
    elif x.dim() == 3:
        assert not torch.is_complex(
            x
        ), "Must be 2-D complex-valued or 3-D real-valued with layered I/Q"
        assert (
            x.shape[1] == 2
        ), "Must be 2-D complex-valued or 3-D real-valued with layered I/Q"

    fr_module.eval()
    
    # Move module and data to execution device
    fr_module, x = fr_module.to(args.device), x.to(args.device)

    # If input dataset has fewer samples than batch_size
    if args.batch_size > x.shape[0]:
        return fr_module(x).detach().cpu()

    temp = fr_module(x[:2])
    y_pred = torch.zeros(x.shape[0], temp.shape[1])

    for i in range(0, x.shape[0], args.batch_size):
        y_pred[i : i + args.batch_size] = (
            fr_module(x[i : i + args.batch_size]).detach().cpu()
        )

    return y_pred
