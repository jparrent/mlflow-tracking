import importlib

def create_data_loader(loader_name):
    data_loader_module = importlib.import_module(f"{__package__}.{loader_name}")
    loader_class = getattr(data_loader_module, loader_name)  # Get the specific loader class based on the name
    return loader_class