from re import U
from fastapi import Depends, FastAPI, File, UploadFile
import pandas as pd
from pycaret.classification import automl
from src.classes.AutoML import Automl
from src.utils.base_encoder import file_to_base64
from src.utils.eval_models import evatuale_performance
import hashlib
import json
from typing import List
from sqlalchemy.orm import Session
from src.database.database import SessionLocal, init_db
from src.models.File import File as DBFile
from src.models.Result import Result as DBResult

app = FastAPI()


def json_2_sha256_key(json_data):
    json_string = json.dumps(json_data)
    sha256_key = hashlib.sha256(json_string.encode()).hexdigest()
    return sha256_key


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/", response_description="Root")
def root():
    return {"Status": "Connected"}


init_db()


@app.post("/updload")
async def upload_file(
    files: List[UploadFile] = File(...), db: Session = Depends(get_db)
):
    for file in files:
        file_data = file.file.read()
        db_file = DBFile(name=file.filename, content=file_data.decode("utf-8"))
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
    return {"detail": f"{len(files)} files uploaded successfully"}


@app.post("/select_model")
def select_model(
    file: UploadFile = File(...),
    n: int = 1,
    metric: str = "Accuracy",
    db: Session = Depends(get_db),
):
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

    keys_dict = {}
    for data in data_2_save:
        key = json_2_sha256_key(data)
        keys_dict[data["model"]] = key

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
    res_json = {"keys": keys_dict, "data": data_2_save, "results": metrics_res}
    print("********** Saving data to Database *********")

    for model, result_data in res_json["results"].items():
        print(model)
        res_instance = DBResult(
            key=res_json["keys"][model],
            model=model,
            accuracy=result_data.get("acc"),
            cross_val=result_data.get("cross_val"),
            roc_score=result_data.get("roc_score"),
            pickle=res_json["data"][0]["pickle"]["data"],
            api=res_json["data"][0]["api"]["data"],
        )
        db.add(res_instance)
        db.commit()
        db.refresh(res_instance)

    return res_json
