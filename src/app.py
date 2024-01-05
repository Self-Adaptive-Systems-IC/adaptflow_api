from fastapi import FastAPI, File, UploadFile
import pandas as pd
from src.classes.AutoML import Automl
from src.utils.base_encoder import file_to_base64
from src.utils.eval_models import evatuale_performance

app = FastAPI()


@app.get("/", response_description="Root")
def root():
    return {"Status": "Connected"}


@app.post("/select_model")
def select_model(file: UploadFile = File(...), n: int = 1, metric: str = "Accuracy"):
    """Select and tune machine learning models based on the provided dataset.

    Args:
        file (UploadFile, optional): Dataset file in CSV format.
        n (int, optional): Number of models to be selected.
        Defaults to 1.
        metric (str, optional): Metric used to order the models.
        Defaults "Accuracy".

    Returns:
        dict: A dictionary containing information about the selected and
        tuned models.
            The dictionary has the following structure:
            {
                'data': [
                    {
                        'dataset': str,
                        'model': str,
                        'pickle': {'name': str,'data': str (base64-encoded)},
                        'api': {'name': str,'data': str (base64-encoded)}
                    },
                    # Additional entries for each selected and tuned model
                ]
            }
    """
    try:
        print(file.content_type)
        if file.content_type != "text/csv":
            return {"error": "000", "message": "Invalid file type"}
        contents = file.file
        df = pd.read_csv(contents)
    except Exception as e:
        return {"error": "001", "message": str(e)}

    dataset_filename = file.filename
    dataset_filename = "" if dataset_filename is None else dataset_filename

    print("********** Starting AutoML **********")
    automl = Automl(df, dataset_filename, metric=metric)

    print(f"-> Select {n} models")
    aux_n = 2 if n == 1 else n
    models = automl.find_best_models(aux_n)
    if n == 1:
        models = [models[0]]

    print("-> Tunning the models")
    tuned_models = []
    for model in models:
        tuned_model = automl.tuning_model(model)
        tuned_models.append(tuned_model)

    print("-> Generating the files")
    data_2_save = []
    base_port = 5000
    commands = []

    for i, model in enumerate(tuned_models):
        print(i)
        dataset_name = dataset_filename.split(".")[0]
        filename = f"{dataset_name}_{model}"

        api_file = f"./tmp/apis/{filename}"

        command = automl.generate_api(model, api_name=api_file, port=base_port + i)
        commands.append(command)

        model_file = automl.save_model(model)

        pickle_base64_string = file_to_base64(f"{model_file}.pkl")
        api_base64_string = file_to_base64(f"{api_file}.py")

        values_2_save = {
            "dataset": dataset_filename,
            "model": model.get_model_name(),
            "pickle": {"name": f"{filename}_pkl", "data": pickle_base64_string},
            "api": {"name": f"{filename}_api", "data": api_base64_string},
        }
        data_2_save.append(values_2_save)

    metrics_res = {}

    for model in tuned_models:
        X_train, y_train, X_test, y_test, df = automl.eval_model(model)
        acc, cros_val, roc_score = evatuale_performance(
            X_train, y_train, X_test, y_test, model.get_estimator()
        )
        metrics_res[model.get_model_name()] = {
            "acc": acc,
            "cross_val": cros_val,
            "roc_score": roc_score,
        }

    print("********** Finishing AutoML **********")
    return {"data": data_2_save, "results": metrics_res}
