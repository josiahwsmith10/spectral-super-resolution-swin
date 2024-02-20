import os
import torch
import modules


def set_device(args):
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def model_parameters(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    return num_params


def save(model, optimizer, scheduler, args, epoch, module_type):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": args,
    }
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if not os.path.exists(os.path.join(args.output_dir, module_type)):
        os.makedirs(os.path.join(args.output_dir, module_type))
    cp = os.path.join(args.output_dir, module_type, "last.pth")
    fn = os.path.join(args.output_dir, module_type, "epoch_" + str(epoch) + ".pth")
    torch.save(checkpoint, fn)
    torch.save(checkpoint, cp)


def load(checkpoint_path, device=torch.device("cuda")):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint["args"]

    # Change args to use defaults for cResFreq from paper
    if "skipfreq" in args.output_dir:
        args.fr_module_type = "cresfreq"
        args.fr_kernel_out = 18
        args.fr_out_padding = 0
        args.optim_type = "adam"
        args.loss_fn = "l2"
    elif "mmpnorm" in args.output_dir:
        args.normalization = "min-max"

    if "fr_dropout" not in vars(args):
        args.fr_dropout = 0.0

    if "fr_optional_relu" not in vars(args):
        args.fr_optional_relu = False

    if device == torch.device("cpu"):
        args.use_cuda = False
    args.device = device
    
    model = modules.select_model(args)
    model.load_state_dict(checkpoint["model"])
    optimizer, scheduler = set_optim(args, model)
    if checkpoint["scheduler"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, scheduler, args, checkpoint["epoch"]


def set_optim(args, module):
    if args.optim_type.lower() == "adam":
        optimizer = torch.optim.Adam(module.parameters(), lr=args.lr_fr)
    elif args.optim_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(module.parameters(), lr=args.lr_fr)
    elif args.optim_type.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(module.parameters(), lr=args.lr_fr, alpha=0.9)
    else:
        raise (ValueError("Unexpected optimizer type"))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=7, factor=0.5, verbose=True
    )
    return optimizer, scheduler


def print_args(logger, args):
    message = ""
    for k, v in sorted(vars(args).items()):
        message += "\n{:>30}: {:<30}".format(str(k), str(v))
    logger.info(message)

    args_path = os.path.join(args.output_dir, "run.args")
    with open(args_path, "wt") as args_file:
        args_file.write(message)
        args_file.write("\n")
