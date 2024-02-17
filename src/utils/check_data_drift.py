import pandas as pd
import json

from evidently.report import Report
from evidently.metrics import DataDriftTable
from evidently.metrics import DatasetDriftMetric

def check_data_drift(reference, current):
    data_drift_dataset_report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable(),
    ])

    ref = reference
    cur = current

    data_drift_dataset_report.run(reference_data=ref, current_data=cur)
    data_drift_dataset_report

    report_json = json.loads(data_drift_dataset_report.json())['metrics'][1]['result']

    res = {}

    res['number_of_columns'] = report_json['number_of_columns']
    res['number_of_drifted_columns'] = report_json['number_of_drifted_columns']
    res['mean_drifted_score'] = report_json['share_of_drifted_columns']
    res['dataset_drift'] = report_json['dataset_drift']

    cols = {}
    drift_by_columns = report_json['drift_by_columns']
    for column_name, column_info in drift_by_columns.items():
        drift_detected = column_info['drift_detected']
        drift_score = column_info['drift_score']
        cols[column_name] = {'drift_detected': drift_detected, 'drift_score': drift_score}
        
    res['columns'] = cols
    return res