import numpy as np

import util
from classical import Periodogram, MUSIC, OMP, FISTA


def is_ml_method(method):
    if (
        method.lower() == "periodogram"
        or method.lower() == "music"
        or method.lower() == "omp"
    ):
        return False
    else:
        return True


def is_ml_model(model):
    return util.model_parameters(model) > 0


def set_method_type(method, methods):
    if "cvswinfreq" in method:
        method_kind = "cvswinfreq"
    elif "swinfreq" in method:
        method_kind = "swinfreq"
    elif "cresfreq" in method:
        return "cresfreq"
    else:
        return method

    # If using method multiple times
    cnt = 1
    method_kind_new = method_kind
    while method_kind_new in methods:
        method_kind_new = f"{method_kind}_v{cnt}"
        cnt += 1

    return method_kind_new


def create_model(method, args):
    print(f"Creating method={method}")
    xgrid = np.linspace(-0.5, 0.5, args.fr_size, endpoint=False)
    if method.lower() == "periodogram":
        model = Periodogram(xgrid)
    elif method.lower() == "music":
        model = MUSIC(
            xgrid,
            m=args.music_m,
            source_number_method=args.source_number_method,
            param=args.source_number_param,
        )
    elif method.lower() == "omp":
        model = OMP(
            signal_dim=args.signal_dim,
            fr_size=args.fr_size,
            m=args.music_m,
            source_number_method=args.source_number_method,
            param=args.source_number_param,
        )
    elif method.lower() == "fista":
        model = FISTA(
            signal_dim=args.signal_dim,
            fr_size=args.fr_size,
        )
    else:
        # method should be a path to a checkpoint
        # load model
        model, _, _, _, _ = util.load(checkpoint_path=method, device=args.device)

    return model


def create_methods(args):
    args.models = {}
    args.methods = []

    for method in args.method_list:
        # Create the model
        model = create_model(method, args)

        # Create short method name
        method_kind_new = set_method_type(method, args.methods)
        args.methods.append(method_kind_new)

        # Set model to methods dictionary
        args.models[method_kind_new] = model
