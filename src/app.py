from fastapi import Depends, FastAPI, File, UploadFile
import pandas as pd
from typing import List
from sqlalchemy.orm import Session

from src.classes.AutoML import Automl

from src.utils.converter import file_to_base64, json_2_sha256_key
from src.utils.eval_models import evatuale_performance

from src.database.config import SessionLocal, init_db

from src.database.models.File import File as DBFile
from src.database.models.Result import Result as DBResult

from src.utils.automl_handler import train_model, tune_models, generate_files

from src.utils.check_data_drift import check_data_drift

app = FastAPI()

# Initialize the database
init_db()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/", response_description="Root")
def root():
    return {"Status": "Connected"}


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
    models = train_model(automl, n)

    print("-> Tunning the models")
    tuned_models = tune_models(automl, models)

    print("-> Generating the files")
    dataset_name = dataset_filename.split(".")[0]
    print(dataset_name)
    data_2_save = generate_files(automl, tuned_models, dataset_name)

    print("********** Finishing AutoML **********")

    print("-> Generating the keys for each json")
    keys_dict = {}
    for data in data_2_save:
        key = json_2_sha256_key(data)
        keys_dict[data["model"]] = key

    print("-> Generating the metrics")
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

    res_json = {"keys": keys_dict, "data": data_2_save, "results": metrics_res}
    print("-> Saving data to Database")

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

    print("***** FINISHED *****")

    return res_json


@app.get("/get_model")
def get_model(key: str, db: Session = Depends(get_db)):
    print("key:", key)
    result = db.query(DBResult).filter(DBResult.key == key).first()
    result = result.json()
    return {"result": result}


@app.get("/get_metrics")
def get_metrics(key: str, db: Session = Depends(get_db)):
    result = db.query(DBResult).filter(DBResult.key == key).first()
    result = result.get_metrics()
    return result


@app.post("/check_drift")
def check_drift(file_reference: UploadFile = File(...),file_current: UploadFile = File(...)):
    try:
        print(file_reference.content_type)
        if file_reference.content_type != "text/csv":
            return {"error": "000", "message": "Invalid file type"}
        contents = file_reference.file
        df_reference = pd.read_csv(contents)
    except Exception as e:
        return {"error": "001", "message": str(e)}
    
    try:
        print(file_current.content_type)
        if file_current.content_type != "text/csv":
            return {"error": "000", "message": "Invalid file type"}
        contents = file_current.file
        df_current = pd.read_csv(contents)
    except Exception as e:
        return {"error": "001", "message": str(e)}
    
    return check_data_drift(df_reference,df_current)