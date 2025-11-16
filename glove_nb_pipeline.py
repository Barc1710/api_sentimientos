import pickle
import numpy as np
import re
from gensim.utils import simple_preprocess
import os

# Parámetros (deben coincidir con el entrenamiento)
EMBEDDING_DIM = 100 
OUTPUT_DIR = 'models'

class GloVeNBPipeline:
    def __init__(self, classifier_path, embeddings_path):
        self.classifier = self.load_model(classifier_path)
        self.embeddings_index = self.load_model(embeddings_path)

    def load_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: No se encontró el archivo del modelo en {path}")
        with open(path, 'rb') as file:
            return pickle.load(file)

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r"[^\w\s']", ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def vectorize_text(self, text):
        """Vectoriza un solo texto usando la lógica GloVe de promedio de palabras."""
        text_clean = self.clean_text(text)
        tokens = simple_preprocess(text_clean)
        
        vectors = [self.embeddings_index[token] 
                   for token in tokens if token in self.embeddings_index]
        
        if vectors:
            # Calcular la media de los vectores (misma lógica que en BLOC 9)
            vector = np.mean(vectors, axis=0)
        else:
            # Vector de ceros si no hay palabras conocidas
            vector = np.zeros(EMBEDDING_DIM)
            
        # El clasificador espera una matriz 2D (1 muestra x N características)
        return vector.reshape(1, -1) 

    def predict(self, text: str):
        X_vec = self.vectorize_text(text)
        prediction = self.classifier.predict(X_vec)[0]
        
        sentiment = "Positivo" if prediction == 1 else "Negativo"
        
        return {
            "prediction": int(prediction),
            "sentiment": sentiment,
            "text": text
        }