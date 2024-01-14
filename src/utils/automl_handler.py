from src.classes.AutoML import Automl
from src.utils.converter import file_to_base64
import os

PATH = os.getcwd()
BASE_FOLDER = f"{PATH}/tmp"
API_FOLDER = "files"


def train_model(automl: Automl, n: int = 1) -> list:
    aux_n = 2 if n == 1 else n
    models = automl.find_best_models(aux_n)
    if n == 1:
        models = [models[0]]
    return models


def tune_models(automl: Automl, models: list) -> list:
    tuned_models = []
    for model in models:
        tuned_model = automl.tuning_model(model)
        tuned_models.append(tuned_model)
    return tuned_models


def generate_files(automl: Automl, tuned_models: list, dataset_name: str):
    data_2_save = []
    base_port = 5000

    for i, model in enumerate(tuned_models):
        filename = f"{dataset_name}_{model}"

        api_file_path = f"{BASE_FOLDER}/{API_FOLDER}/{filename}"

        automl.generate_api(model, api_name=api_file_path, port=base_port + i)

        pickle_file_path = automl.save_model(model)

        api_base64 = file_to_base64(f"{api_file_path}.py")
        pickle_base64 = file_to_base64(f"{pickle_file_path}.pkl")

        values_2_save = {
            "dataset": dataset_name,
            "model": model.get_model_name(),
            "pickle": {"name": f"{filename}_pkl", "data": pickle_base64},
            "api": {"name": f"{filename}_api", "data": api_base64},
        }

        data_2_save.append(values_2_save)

    return data_2_save
