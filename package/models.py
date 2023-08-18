import pickle
import pandas as pd


def read_data(path="data/df_train_selected.csv"):
    return pd.read_csv(path).reset_index(drop=True)


def load_model(path="data/best_lr.pkl"):
    with open(path, 'rb') as model_file:
        return pickle.load(model_file)


def get_model_param(path="data/best_lr.pkl"):
    model = load_model(path)
    return model.get_params()
