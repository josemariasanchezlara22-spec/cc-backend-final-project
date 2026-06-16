from fastapi import FastAPI, UploadFile, File, HTTPException
from google.cloud import aiplatform
import os

# ... (Todo tu código anterior de inicialización y /upload-csv/ se queda igual)

# ==========================================
# 1. EN LÍNEA: PREDICCIÓN POR REGISTRO INDIVIDUAL
# ==========================================
@app.post("/predict-single/")
async def predict_single(data: dict):
    """
    Recibe los datos de UN cliente desde el formulario de React.
    Ejemplo de entrada: {"edad": 34, "ingresos": 45000, "historial": 1}
    """
    try:
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)
        
        # Extraemos los valores del JSON en el orden exacto que tu modelo fue entrenado
        # NOTA: Cambia estas claves por los nombres reales de tus variables
        features = [
            data.get("edad", 0),
            data.get("ingresos", 0),
            data.get("historial", 0)
        ]
        
        # Vertex AI espera una lista de instancias: [[val1, val2, val3]]
        response = endpoint.predict(instances=[features])
        
        return {
            "tipo_prediccion": "registro_individual",
            "probabilidad_impago": response.predictions[0],
            "resultado": "Rechazado" if response.predictions[0] > 0.5 else "Aprobado"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción online: {str(e)}")


# ==========================================
# 2. POR LOTES: PREDICCIÓN MASIVA (BATCH)
# ==========================================
@app.post("/predict-batch/")
async def predict_batch(file: UploadFile = File(...)):
    """
    Recibe un archivo CSV con MILES de clientes, lo manda a Vertex AI,
    procesa en lote y guarda los resultados en el Bucket.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Debes cargar un archivo CSV para el lote.")
        
    try:
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        
        # 1. Subir el archivo de lote a una carpeta temporal en tu Storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        input_blob = bucket.blob(f"batch_inputs/{file.filename}")
        input_blob.upload_from_file(file.file)
        
        gcs_input_uri = f"gs://{BUCKET_NAME}/batch_inputs/{file.filename}"
        gcs_output_uri_prefix = f"gs://{BUCKET_NAME}/batch_outputs/"
        
        # 2. Buscar tu modelo registrado en Vertex AI (Model Registry)
        # Reemplaza 'MODEL_ID' con el ID que te da Vertex AI en su sección de modelos
        model = aiplatform.Model(model_name="MODEL_ID")
        
        # 3. Disparar el trabajo por lotes en Vertex AI
        batch_job = model.batch_predict(
            job_display_name=f"batch_scoring_{file.filename}",
            gcs_source=gcs_input_uri,
            gcs_destination_prefix=gcs_output_uri_prefix,
            instances_format="csv",
            predictions_format="csv",
            machine_type="n1-standard-4" # Levanta una máquina intermedia para el lote y luego la apaga
        )
        
        return {
            "tipo_prediccion": "lote_masivo",
            "status": "Trabajo enviado a Vertex AI",
            "job_name": batch_job.name,
            "input_origen": gcs_input_uri,
            "carpeta_destino_resultados": gcs_output_uri_prefix
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al iniciar lote en Vertex AI: {str(e)}")