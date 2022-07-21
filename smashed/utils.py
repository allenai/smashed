import importlib


def requires(module_name: str):
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ImportError(f'{module_name} is required for this module')
