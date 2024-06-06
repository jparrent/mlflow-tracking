import importlib

def create_feature_engineer(feature_engineer_name):
    feature_engineer_module = importlib.import_module(f"{__package__}.{feature_engineer_name}")
    feature_engineer_class = getattr(feature_engineer_module, feature_engineer_name)  # Get the specific feature_engineer class based on the name
    return feature_engineer_class