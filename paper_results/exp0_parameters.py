import util
from .util import is_ml_method


def get_model_parameters(args):
    parameters = {}
    for method, model in args.models.items():
        parameters[method] = util.model_parameters(model)

    return parameters


def print_parameters(args, parameters):
    print("\nNumber of parameters for each method:")
    for method in args.methods:
        if is_ml_method(method):
            print(f"{method} : {parameters[method]}")
    print("")


def experiment0(args):
    parameters = get_model_parameters(args)

    print_parameters(args, parameters)
