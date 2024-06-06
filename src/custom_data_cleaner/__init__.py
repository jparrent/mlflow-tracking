import importlib

def create_data_cleaner(cleaner_name):
    data_cleaner_module = importlib.import_module(f"{__package__}.{cleaner_name}")
    cleaner_class = getattr(data_cleaner_module, cleaner_name)  # Get the specific cleaner class based on the name
    return cleaner_class