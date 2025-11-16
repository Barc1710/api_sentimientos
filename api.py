from fastapi import FastAPI
from pydantic import BaseModel
from glove_nb_pipeline import GloVeNBPipeline
import os
from fastapi.middleware.cors import CORSMiddleware

# Rutas de los archivos (estandarizacion/prediccion)
CLASSIFIER_PATH = os.path.join('models', 'glove_nb_classifier.pkl')
EMBEDDINGS_PATH = os.path.join('models', 'glove_embeddings_index.pkl')

# ... (toda la lógica de cargar el pipeline sigue igual) ...
try:
    pipeline = GloVeNBPipeline(CLASSIFIER_PATH, EMBEDDINGS_PATH)
    print("✅ Modelo GloVe_NB y Embeddings cargados exitosamente para la API.")
except FileNotFoundError as e:
    print(f"❌ Error fatal al cargar el modelo: {e}. Asegúrate de que los archivos .pkl estén en la carpeta 'models'.")
    pipeline = None

app = FastAPI(title="API de Clasificación de Sentimientos - GloVe_NB")

# Esto le da permiso a tu frontend  para que se conecte
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes (para desarrollo)
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST)
    allow_headers=["*"],
)

class TextToClassify(BaseModel):
    text: str
    
@app.get("/")
def read_root():
    return {"message": "API de Clasificación de Sentimientos lista. Usa el endpoint /predict."}

@app.post("/predict")
def predict_sentiment(item: TextToClassify):
    if pipeline is None:
        return {"error": "El modelo no pudo ser cargado. Revise el log de la API."}
        
    # Realizar la predicción
    result = pipeline.predict(item.text)
    
    # Especificación de la Salida
    return {
        "text": result['text'],
        "prediction": result['prediction'], # 1 o 0
        "sentiment": result['sentiment']    # Positivo o Negativo
    }