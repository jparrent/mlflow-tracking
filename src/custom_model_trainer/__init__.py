import importlib

def create_model_trainer(model_trainer_name):
    model_trainer_module = importlib.import_module(f"{__package__}.{model_trainer_name}")
    model_trainer_class = getattr(model_trainer_module, model_trainer_name)  # Get the specific model_trainer class based on the name
    return model_trainer_class