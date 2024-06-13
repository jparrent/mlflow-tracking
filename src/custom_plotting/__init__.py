import importlib

def create_plotting(plotting_name):
    plotting_module = importlib.import_module(f"{__package__}.{plotting_name}")
    plotting_class = getattr(plotting_module, plotting_name)  # Get the specific plotting class based on the name
    return plotting_class
