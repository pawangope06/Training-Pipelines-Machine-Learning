from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model

def run_pipeline(config):
    X, y = load_data(config["data"])
    X_train, X_test, y_train, y_test = preprocess_data(X, y, config["data"])
    model = train_model(X_train, y_train, config["model"])
    evaluate_model(model, X_test, y_test, config["evaluation"])
    return model
