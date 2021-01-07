import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame({"Sigla_UF": pd.Series(["example_value"], dtype="object"), "Cod_Municipio": pd.Series([0.0], dtype="float64"), "Nome_Municipio": pd.Series(["example_value"], dtype="object"), "Cod_Escola": pd.Series([0.0], dtype="float64"), "Nome_Escola": pd.Series(["example_value"], dtype="object"), "Column6": pd.Series(["example_value"], dtype="object"), "TAprov2019_Total": pd.Series([0.0], dtype="float64"), "TAprov2019_1_serie": pd.Series(["example_value"], dtype="object"), "TAprov2019_2_serie": pd.Series(["example_value"], dtype="object"), "TAprov2019_3_serie": pd.Series([0.0], dtype="float64"), "TAprov2019_4_serie": pd.Series(["example_value"], dtype="object"), "Indicador_Rendimento": pd.Series([0.0], dtype="float64"), "SAEB2019_Matematica": pd.Series([0.0], dtype="float64"), "SAEB2019_Lingua Portuguesa": pd.Series([0.0], dtype="float64"), "SAEB2019_Nota Media Padronizada": pd.Series([0.0], dtype="float64")})
output_sample = np.array([0])
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[1], 'model_version': path_split[2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})