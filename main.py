from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from google.cloud import aiplatform
import os
import json

# ==============================================================================
# 1. INICIALIZACIÓN Y CONFIGURACIÓN DE LA API
# ==============================================================================
app = FastAPI(
    title="API de Riesgo Crediticio - MLOps Dinámico",
    description="Backend asíncrono automatizado para re-entrenamiento y predicción en Vertex AI",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CONTADORES EN MEMORIA
estadisticas_operativas = {
    "total_evaluaciones": 0,
    "aprobados": 0,
    "rechazados": 0
}

# ==============================================================================
# 2. CONFIGURACIÓN DE VARIABLES REALES DE GCP
# ==============================================================================
BUCKET_NAME = "am-up-01-credit-riskv2"
PROJECT_ID = "cloud-computing-jm-2026v2"
LOCATION = "us-central1"
ENDPOINT_ID = "6078901272566562816"

aiplatform.init(project=PROJECT_ID, location=LOCATION)


# ==============================================================================
# 3. ENDPOINTS / RUTAS DE LA API
# ==============================================================================

@app.get("/")
def root():
    return {
        "status": "online", 
        "architecture": "event-driven-mlops-async",
        "message": "Backend conectado y optimizado"
    }

@app.get("/metrics/")
def obtener_metricas():
    total = estadisticas_operativas["total_evaluaciones"]
    if total == 0:
        return {"pct_aprobados": 0, "pct_rechazados": 0, "total": 0}
    
    pct_aprobados = round((estadisticas_operativas["aprobados"] / total) * 100, 1)
    pct_rechazados = round((estadisticas_operativas["rechazados"] / total) * 100, 1)
    
    return {
        "pct_aprobados": pct_aprobados,
        "pct_rechazados": pct_rechazados,
        "total": total
    }

# ------------------------------------------------------------------------------
# ENDPOINT NUEVO: Leer predicciones generadas por el proceso por lotes desde GCS
# ------------------------------------------------------------------------------
@app.get("/batch-results/")
def obtener_resultados_batch():
    """
    Busca de manera dinámica el archivo 'prediction.results' más reciente
    dentro de la estructura de carpetas de batch_outputs en Cloud Storage.
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # Listar objetos con el prefijo donde Vertex AI deposita las respuestas
        blobs = bucket.list_blobs(prefix="batch_outputs/")
        
        # Filtrar para encontrar el archivo de resultados real generado
        target_blob = None
        for blob in blobs:
            if "prediction.results-" in blob.name:
                target_blob = blob
                # No rompemos el ciclo para asegurarnos de agarrar el último subido
        
        if not target_blob:
            return {"status": "no_files", "results": [], "message": "No se encontraron archivos de resultados procesados aún."}
            
        # Descargar el contenido JSON Lines en memoria
        content = target_blob.download_as_text()
        lineas = content.splitlines()
        
        parsed_results = []
        for index, line in enumerate(lineas):
            if not line.strip():
                continue
            data = json.loads(line)
            
            # Ajustar según la estructura exacta que arroja tu modelo XGBoost entrenado con Optuna
            # Usualmente viene como {"prediction": [score], "label": [0]} o similar
            instance_data = data.get("instance", {})
            prediction_data = data.get("prediction", {})
            
            # Vertex AI Batch suele retornar las clases en llaves como 'label' o la primera posición de un array
            clase = int(prediction_data[0] if isinstance(prediction_data, list) else prediction_data.get("label", 0))
            probabilidad = float(prediction_data[1] if isinstance(prediction_data, list) else prediction_data.get("probability", 0.0))
            
            parsed_results.append({
                "id": index + 1,
                "monto_solicitado": instance_data.get("loan_amnt", "N/A"),
                "proposito": instance_data.get("purpose", "N/A"),
                "probabilidad_default": round(probabilidad * 100, 2) if probabilidad <= 1 else probabilidad,
                "dictamen": "Rechazado (Riesgo Alto)" if clase == 1 else "Aprobado (Riesgo Bajo)",
                "clase": clase
            })
            
        return {
            "status": "success",
            "archivo_origen": target_blob.name,
            "results": parsed_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer resultados desde GCS: {str(e)}")

# ------------------------------------------------------------------------------
# ENDPOINTS PREVIOS (UPLOAD CSV, SINGLE PREDICT Y BATCH PREDICT)
# ------------------------------------------------------------------------------
@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="El archivo debe ser un formato CSV válido.")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"data/raw/{file.filename}")
        blob.upload_from_file(file.file, content_type=file.content_type)
        return {
            "status": "success",
            "message": f"Archivo '{file.filename}' depositado en el Storage.",
            "event_trigger": "Pipeline de re-entrenamiento detonado en segundo plano",
            "destination": f"gs://{BUCKET_NAME}/data/raw/{file.filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Cloud Storage: {str(e)}")

@app.post("/predict-single/")
async def predict_single(data: dict):
    try:
        endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)
        cliente_instancia = {
            "loan_amnt": int(data.get("loan_amnt", 0)),
            "term": str(data.get("term", " 36 months")),
            "int_rate": float(data.get("int_rate", 0.0)),
            "installment": float(data.get("installment", 0.0)),
            "grade": str(data.get("grade", "B")),
            "sub_grade": str(data.get("sub_grade", "B3")),
            "emp_length": str(data.get("emp_length", "5 years")),
            "home_ownership": str(data.get("home_ownership", "MORTGAGE")),
            "annual_inc": float(data.get("annual_inc", 0.0)),
            "verification_status": str(data.get("verification_status", "Verified")),
            "application_type": str(data.get("application_type", "Individual")),
            "purpose": str(data.get("purpose", "debt_consolidation")),
            "dti": float(data.get("dti", 0.0)),
            "delinq_2yrs": int(data.get("delinq_2yrs", 0)),
            "inq_last_6mths": int(data.get("inq_last_6mths", 0)),
            "open_acc": int(data.get("open_acc", 0)),
            "pub_rec": int(data.get("pub_rec", 0)),
            "revol_bal": int(data.get("revol_bal", 0)),
            "revol_util": float(data.get("revol_util", 0.0)),
            "total_acc": int(data.get("total_acc", 0)),
            "mort_acc": int(data.get("mort_acc", 0)),
            "pub_rec_bankruptcies": int(data.get("pub_rec_bankruptcies", 0)),
            "tax_liens": int(data.get("tax_liens", 0))
        }
        response = endpoint.predict(instances=[cliente_instancia])
        primera_prediccion = response.predictions[0]
        clase = int(primera_prediccion.get("label"))              
        probabilidad = float(primera_prediccion.get("probability"))  
        
        estadisticas_operativas["total_evaluaciones"] += 1
        if clase == 0:
            estadisticas_operativas["aprobados"] += 1
        else:
            estadisticas_operativas["rechazados"] += 1
        
        return {
            "tipo_prediccion": "registro_individual",
            "clase_predicha": clase,
            "probabilidad_impago": probabilidad,
            "resultado": "Rechazado (Riesgo Alto)" if clase == 1 else "Aprobado (Riesgo Bajo)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Vertex AI Online: {str(e)}")

@app.post("/predict-batch/")
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Debes cargar un archivo CSV válido para el lote.")
    try:
        contents = await file.read()
        lines = contents.decode("utf-8").splitlines()
        if len(lines) < 2:
            raise HTTPException(status_code=400, detail="El archivo CSV está vacío.")
            
        headers = [h.strip() for h in lines[0].split(",")]
        jsonl_lines = []
        for line in lines[1:]:
            if not line.strip():
                continue
            values = [v.strip() for v in line.split(",")]
            row_dict = {}
            for header, val in zip(headers, values):
                if val.replace('.', '', 1).isdigit():
                    row_dict[header] = float(val) if '.' in val else int(val)
                else:
                    row_dict[header] = val.replace('"', '')
            jsonl_lines.append(json.dumps(row_dict))
            
        jsonl_content = "\n".join(jsonl_lines)
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        filename_base = file.filename.rsplit('.', 1)[0]
        jsonl_filename = f"{filename_base}.jsonl"
        
        input_blob = bucket.blob(f"batch_inputs/{jsonl_filename}")
        input_blob.upload_from_string(jsonl_content, content_type="application/jsonl")
        
        gcs_input_uri = f"gs://{BUCKET_NAME}/batch_inputs/{jsonl_filename}"
        gcs_output_uri_prefix = f"gs://{BUCKET_NAME}/batch_outputs/"
        
        endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)
        modelos_desplegados = endpoint.list_models()
        if not modelos_desplegados:
            raise Exception("El Endpoint no cuenta con ningún modelo desplegado.")
        
        model_id_dinamico = modelos_desplegados[0].model
        model = aiplatform.Model(model_name=model_id_dinamico)
        
        batch_job = model.batch_predict(
            job_display_name=f"batch_scoring_dynamic_{filename_base}",
            gcs_source=gcs_input_uri,
            gcs_destination_prefix=gcs_output_uri_prefix,
            instances_format="jsonl",     
            predictions_format="jsonl",   
            machine_type="n1-standard-4",
            sync=False
        )
        return {
            "tipo_prediccion": "lote_masivo",
            "status": "Trabajo de predicción masiva enviado con éxito a Vertex AI",
            "modelo_utilizado_id": model_id_dinamico,
            "job_id": batch_job.name,
            "mensaje": "Se convirtió el CSV a JSON Lines automáticamente.",
            "input_origen": gcs_input_uri,
            "carpeta_destino_resultados": gcs_output_uri_prefix
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Vertex AI Batch Dinámico: {str(e)}")