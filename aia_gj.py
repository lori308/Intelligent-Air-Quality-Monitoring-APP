import streamlit as st
import base64
import psutil
import re
import requests
import pandas as pd
import os
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
import fitz  # PyMuPDF
import json
import folium 
from streamlit.components.v1 import html as st_html
from functools import lru_cache
from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.cohere import CohereEmbedding  
from scipy.spatial.distance import cosine
import chromadb
import numpy as np
from typing import List
import cohere
from dotenv import load_dotenv
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# CARICAMENTO DELLE VARIABILI D'AMBIENTE + SICURO
load_dotenv()

# CONFIGURAZIONE PAGINA
st.set_page_config(
    page_title="Monitoraggio Qualità dell'Aria", 
    page_icon="🌬️", 
    layout="wide"
)

#CONFIGURAZIONI GENERALI
#API_KEY'S & TOKEN

#VERSIONE CON CHIAVI IN MODALITA' SICURA

OPENWEATHER_FORECAST_ENDPOINT = "https://api.openweathermap.org/data/2.5/forecast";
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY");
COHERE_API_KEY = os.environ.get("COHERE_API_KEY");
COHERE_API_ENDPOINT = "https://api.cohere.ai/v1/chat";

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN");
GITHUB_AZURE_ENDPOINT = os.environ.get("GITHUB_AZURE_ENDPOINT");
GITHUB_MODEL_NAME = os.environ.get("GITHUB_MODEL_NAME", "gpt-4o");


#VERSIONE IN MODALITA' NON SICURA ONLY VISUALIZZAZIONE 
# (ATTENZIONE A NON DIVULGARE LE CHIAVI API)




#CARTELLE IMPORTANTI
KNOWLEDGE_BASE_DIR = "knowledge_base"
VECTOR_DB_DIR = "vector_db"
EMBEDDINGS_CACHE_DIR = "embeddings_cache" 

#REGIONI DISPONIBILI QUALITA' ARIA
VALID_REGIONS = {
    "Piemonte", "Lombardia", "Lazio", "Puglia", "Campania",
    "Veneto", "Emilia-Romagna", "Toscana", "Sicilia", "Liguria",
    "Abruzzo", "Basilicata", "Calabria", "Friuli-Venezia Giulia", "Marche",
    "Molise", "Sardegna", "Trentino-Alto Adige", "Umbria", "Valle d'Aosta", 
}
INPUT_PATTERN = re.compile(r'^[a-zA-Z\s\']{2,50}$')


# CONFIGURAZIONE LOGGING
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# CCREAZIONE DIRECTORY GENERALI
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_CACHE_DIR, exist_ok=True)  # <-- Crea la directory per il cache


# DEFINIZIONE COLORI E DESCRIZIONE PER OGNI VALORE DI AQI
AQI_COLORS = ["#00ADEF", "#00A550", "#F7941E", "#FFD700", "#FF0000", "#800080"]
AQI_LABELS = ["1", "2", "3", "4", "5", "6"]
AQI_DESCRIPTIONS = ["Buona", "Discreta", "Nella Media", "Scadente", "Molto Scadente", "Estremamente Scadente"]


# GENERAZIONE ID UNIVOCI PER I TESTI
def generate_text_id(text):
    """Genera un ID univoco basato sul contenuto del testo"""
    import hashlib
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# FUNZIONE PER SALVARE GLI EMBEDDING LOCALMENTE
def save_embedding_locally(text_id, embedding):
    """Salva un vettore di embedding nel file system locale"""
    try:
        # CONVERTE L'EMBEDDING IN LISTA SE E' UN ARRAY NUMPY
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        # Dati da salvare
        data = {
            "id": text_id,
            "embedding": embedding,
            "timestamp": datetime.now().isoformat()
        }
        # Percorso del file
        file_path = os.path.join(EMBEDDINGS_CACHE_DIR, f"{text_id}.json")
        # Salva il file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info(f"Embedding salvato con successo: {text_id[:8]}...")
        return True
    except Exception as e:
        logger.error(f"Errore nel salvataggio dell'embedding: {str(e)}")
        return False

# FUNZIONE CHE CARICA GLI EMBEDDING DAL CACHE LOCALE
def load_embedding_from_cache(text_id):
    """Carica un embedding dal cache locale se esiste"""
    try:
        file_path = os.path.join(EMBEDDINGS_CACHE_DIR, f"{text_id}.json")
        
        # Verifica se il file esiste
        if not os.path.exists(file_path):
            return None
            
        # Carica il file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Estrai e restituisci l'embedding
        embedding = data.get("embedding")
        if embedding:
            return np.array(embedding)
        return None
    except Exception as e:
        logger.error(f"Errore nel caricamento dell'embedding dal cache: {str(e)}")
        return None

# CLASSE PERSONALIZZATA PER L'EMBEDDING COHERE CON CLASSE LOCALE
class CachedCohereEmbedding(CohereEmbedding):
    """Estende CohereEmbedding per salvare/caricare embedding dal cache locale"""
    
    def _handle_embedding_with_cache(self, text_or_texts, batch_mode=False):
        """Gestione centralizzata della cache per embedding singoli e batch"""
        
        if not batch_mode:
            text = text_or_texts
            text_id = generate_text_id(text)
            cached = load_embedding_from_cache(text_id)
            
            if cached is not None:
                logger.info(f"Embedding caricato dal cache: {text_id[:8]}...")
                return cached.tolist()
                
            
            embedding = super().get_text_embedding(text)
            save_embedding_locally(text_id, embedding)
            return embedding
            
        
        texts = text_or_texts
        results = [None] * len(texts)
        to_process = []
        id_map = {}  # Mapping tra indici
        
        
        for i, text in enumerate(texts):
            text_id = generate_text_id(text)
            cached = load_embedding_from_cache(text_id)
            
            if cached is not None:
                results[i] = cached.tolist()
            else:
                to_process.append(text)
                id_map[len(to_process)-1] = (i, text_id)
        
        
        if not to_process:
            return results
            
        # ALTRIMENTI PROCESSIAMO IL BATCH
        new_embeddings = super().get_text_embedding_batch(to_process)
        
        # AGGIORNAMENTO PROCESSI IN CACHE
        for i, emb in enumerate(new_embeddings):
            orig_idx, text_id = id_map[i]
            save_embedding_locally(text_id, emb)
            results[orig_idx] = emb
            
        return results
    
    def get_text_embedding(self, text: str) -> List[float]:
        return self._handle_embedding_with_cache(text, batch_mode=False)
    
    def get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        return self._handle_embedding_with_cache(texts, batch_mode=True)

# CONFIGURAZIONE DEL MODELLO DI EMBEDDING COHERE CON CACHE LOCALE
def setup_embedding_model():
    """Configura l'embedding model di Cohere con cache locale"""
    try:
        logger.info("Configurazione embedding Cohere con cache locale...")
        
        # CREAZIONE MODELLO COHERE CON API KEY E CACHE 
        embed_model = CachedCohereEmbedding(
            cohere_api_key=COHERE_API_KEY,
            model_name="embed-multilingual-v3.0",
            truncate="END"
        )
        
        # IMPOSTA GLOBALMENTE L'EMBEDDING MODEL
        Settings.embed_model = embed_model
        
        logger.info("✅ Modello Cohere configurato correttamente con cache locale")
        return embed_model
        
    except Exception as e:
        logger.error(f"❌ Errore configurazione Cohere: {str(e)}")
        st.error("""
            Errore configurazione embedding Cohere. Verifica:
            1. La chiave API Cohere è valida
            2. Il pacchetto 'llama-index-embeddings-cohere' è installato
            3. La connessione internet è attiva
        """)
        raise  

# FUNZIONI PER IL DATABASE VETTORIALE CHROMADB
def create_chroma_index(documents):
    """Crea un nuovo indice ChromaDB dai documenti forniti"""
    try:
        # CONFIGURA IL MODELLO DI EMBEDDING
        if not Settings.embed_model:
            setup_embedding_model()
        
        chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        collection_name = "air_quality_collection"
        chroma_collection = chroma_client.get_or_create_collection(name=collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        parser = SimpleNodeParser.from_defaults(chunk_size=512)
        nodes = parser.get_nodes_from_documents(documents)
        
        index = VectorStoreIndex(
            nodes,
            vector_store=vector_store
        )
        
        logger.info(f"Indice ChromaDB creato con {len(documents)} documenti")
        return index
    except Exception as e:
        logger.error(f"Errore nella creazione dell'indice ChromaDB: {str(e)}")
        return None
    
def load_chroma_index():
    """Carica un indice ChromaDB esistente"""
    try:
        if not Settings.embed_model:
            setup_embedding_model()
        
        if not os.path.exists(VECTOR_DB_DIR):
            logger.warning("Directory ChromaDB non trovata")
            return None
        
        chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        collection_name = "air_quality_collection"
        try:
            chroma_collection = chroma_client.get_collection(name=collection_name)
        except ValueError:
            logger.warning(f"Collezione '{collection_name}' non trovata")
            return None
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        logger.info("Indice ChromaDB caricato con successo")
        return index
    except Exception as e:
        logger.error(f"Errore nel caricamento dell'indice ChromaDB: {str(e)}")
        return None
    
# FUNZIONI PER PREPARARE E AGGIUNGERE DOCUMENTI AL DATABASE 
def prepare_air_quality_document(air_data, location, region): 
    """Prepara un documento per l'indice vettoriale dai dati sulla qualità dell'aria"""
    try:
        components = air_data['list'][0]['components']
        aqi = air_data['list'][0]['main']['aqi']
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Crea un testo strutturato con i dati sulla qualità dell'aria
        content = f"""
        Dati sulla qualità dell'aria per {location}, {region} rilevati il {timestamp}.
        
        Indice di qualità dell'aria (AQI): {aqi} - {AQI_DESCRIPTIONS[aqi-1]}.
        
        Concentrazioni degli inquinanti:
        - PM2.5: {components.get('pm2_5', 0)} μg/m³
        - PM10: {components.get('pm10', 0)} μg/m³
        - NO2: {components.get('no2', 0)} μg/m³
        - SO2: {components.get('so2', 0)} μg/m³
        - O3: {components.get('o3', 0)} μg/m³
        - CO: {components.get('co', 0)} μg/m³
        
        Analisi rispetto ai limiti:
        - Limite OMS per PM2.5: 5 μg/m³ (media annuale)
        - Limite OMS per PM10: 15 μg/m³ (media annuale)
        - Limite OMS per NO2: 10 μg/m³ (media annuale)
        - Limite OMS per SO2: 40 μg/m³ (media giornaliera)
        - Limite OMS per O3: 100 μg/m³ (max giornaliero 8 ore)
        - Limite OMS per CO: 4 mg/m³ (media su 8 ore)
        """
        
        # CREA DOCUMENTO UTILIZZANDO LLAMAINDEX
        document = Document(
            text=content,
            metadata={
                "location": location,
                "region": region,
                "timestamp": timestamp,
                "aqi": aqi,
                "type": "air_quality_data",
                "pm25": components.get('pm2_5', 0),
                "pm10": components.get('pm10', 0),
                "no2": components.get('no2', 0),
                "so2": components.get('so2', 0),
                "o3": components.get('o3', 0),
                "co": components.get('co', 0)
            }
        )
        
        return document
    except Exception as e:
        logger.error(f"Errore nella preparazione del documento: {str(e)}")
        return None
# FUNZIONE CHE AGGIUNGE I DOCUMENTI ALL'INDICE VETTORIALE
def add_air_quality_to_index(index, air_data, location, region):
    """Aggiunge i dati sulla qualità dell'aria all'indice vettoriale ChromaDB"""
    try:
        if index is None:
            logger.warning("Indice vettoriale ChromaDB non disponibile, impossibile aggiungere i dati")
            return False
            
        document = prepare_air_quality_document(air_data, location, region)
        if document is None:
            return False
            
        index.insert(document)
        
        logger.info(f"Dati sulla qualità dell'aria per {location}, {region} aggiunti all'indice vettoriale ChromaDB")
        return True
    except Exception as e:
        logger.error(f"Errore nell'aggiunta dei dati all'indice ChromaDB: {str(e)}")
        return False

# STILE CSS PER IL FRONTEND DELLE PAGINE
def load_css():
    """Carica CSS da file esterno"""
    css_file_path = "styles.css"
    
    try:
        with open(css_file_path, "r", encoding="utf-8") as f:
            css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
            logger.info(f"CSS caricato correttamente da {css_file_path}")
    except Exception as e:
        logger.error(f"Errore nel caricamento del CSS: {str(e)}")
        

# INIZIALIZZAZIONE DELLA SESSIONE
def init_session_state(defaults):
    """Inizializza variabili di sessione con valori predefiniti"""
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# FUNZIONE GENERICA PER RICHIESTE API
def fetch_api_data(endpoint, params=None, base_url=None, api_key=None):
    """Funzione generica per le richieste API con caching"""
    try:
        url = f"{base_url or ''}{endpoint}"
         
        if api_key and params:
            params["appid"] = api_key
        elif api_key:
            params = {"appid": api_key}
            
        cache_key = json.dumps(params) if params else "no_params"
        
        @lru_cache(maxsize=100)
        def cached_fetch(url, params_str):
            p = json.loads(params_str) if params_str != "no_params" else None
            response = requests.get(
                url, 
                params=p,
                headers={"User-Agent": "AirQualityApp/1.0"}, 
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        
        return cached_fetch(url, cache_key)
    except Exception as e:
        logger.error(f"Errore nella richiesta a {endpoint}: {str(e)}")
        return None

# FUNZIONI DI SUPPORTO
def validate_input(text: str, field_name: str) -> bool:
    """Valida l'input dell'utente"""
    if not text or not INPUT_PATTERN.match(text):
        st.error(f"{field_name} contiene caratteri non validi o lunghezza errata")
        return False
    return True

def normalize_region(region: str, location: str) -> str:
    """Normalizza il nome della regione"""
    if not region:
        if location.lower() in ("roma", "fiumicino"):
            return "Lazio"
        elif location.lower() in ("milano", "pavia"):
            return "Lombardia"
        return "Regione non specificata"
    region = region.strip().title()
    return region if region in VALID_REGIONS else "Regione non valida"

# FUNZIONE CHE CONVERTE TIPI NUMPY IN TIPI PYTHON STANDARD
def numpy_to_python(obj):
    """Converte tipi NumPy in tipi Python standard per la serializzazione JSON."""
    import numpy as np
    
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(i) for i in obj]
    else:
        return obj

#FUNZIONE CHE OTTIENE LE COORDINATE TRAMITE OPENSTREETMAP
def get_coordinates_osm(location: str, region: str) -> tuple:
    """Ottiene le coordinate geografiche tramite OpenStreetMap"""
    try:
        params = {"q": f"{location}, {region}", "format": "json", "limit": 1}
        data = fetch_api_data("https://nominatim.openstreetmap.org/search", params)
        
        if not data:
            st.error("Località non trovata")
            return None, None
        
        return float(data[0]['lat']), float(data[0]['lon'])
    except Exception as e:
        st.error(f"Errore durante il recupero delle coordinate: {str(e)}")
        return None, None

# FUNZIONE CHE OTTIENE I DATI RELATIVI SULLA QUALITA' DELL'ARIA 
def get_air_pollution(lat: float, lon: float) -> dict:
    """Ottiene i dati sulla qualità dell'aria attuali"""
    try:
        params = {"lat": lat, "lon": lon}
        data = fetch_api_data(
            "http://api.openweathermap.org/data/2.5/air_pollution", 
            params, 
            api_key=OPENWEATHER_API_KEY
        )
        
        if not data or not data.get('list'):
            st.warning("Nessun dato disponibile per questa località")
            return None
            
        return data
    except Exception as e:
        st.error(f"Errore durante il recupero dati: {str(e)}")
        return None

#FUNZIONE CHE OTTIENE I DATI RELATIVI ALLE PREVISIONI METEO
def get_weather_forecast(lat: float, lon: float) -> dict:
    """Ottiene le previsioni meteo per i prossimi giorni"""
    try:
        params = {"lat": lat, "lon": lon, "units": "metric", "lang": "it"}
        data = fetch_api_data(
            OPENWEATHER_FORECAST_ENDPOINT, 
            params, 
            api_key=OPENWEATHER_API_KEY
        )
        if not data or not data.get('list'):
            st.warning("Nessuna previsione meteo disponibile per questa località")
            return None
            
        # ORGANIZZAZIONE DELLE PREVISIONI PER I PROSSIMI DUE GIORNI
        forecasts = []
        today = datetime.now().date()
        
        for item in data['list']:
            forecast_time = datetime.fromtimestamp(item['dt'])
            forecast_date = forecast_time.date()
            
            days_difference = (forecast_date - today).days
            if 0 <= days_difference <= 1:  # Solo oggi e domani
                forecasts.append({
                    'data': forecast_time,
                    'data_testo': forecast_time.strftime('%d/%m %H:%M'),
                    'giorno': days_difference,
                    'temperatura': item['main']['temp'],
                    'umidita': item['main']['humidity'],
                    'pressione': item['main']['pressure'],
                    'vento_velocita': item['wind']['speed'],
                    'vento_direzione': item['wind']['deg'],
                    'condizioni': item['weather'][0]['description'],
                    'icona': item['weather'][0]['icon'],
                    'probabilita_pioggia': item.get('pop', 0) * 100  # Convertito in percentuale
                })
        return {
            'forecasts': forecasts,
            'city': data['city']['name']
        }
    except Exception as e:
        st.error(f"Errore durante il recupero previsioni meteo: {str(e)}")
        return None

#FUNZIONE CHE OTTIENE I DATI RELATIVI SULLA QUALITA' DELL'ARIA PER I PROSSIMI GIORNI
def get_air_pollution_forecast(lat: float, lon: float) -> dict:
    """Ottiene le previsioni sulla qualità dell'aria per i prossimi giorni"""
    try:
        params = {"lat": lat, "lon": lon}
        data = fetch_api_data(
            "http://api.openweathermap.org/data/2.5/air_pollution/forecast", 
            params, 
            api_key=OPENWEATHER_API_KEY
        )
        if not data or not data.get('list'):
            st.warning("Nessuna previsione sulla qualità dell'aria disponibile")
            return None
            
        # ORGANIZZAZIONE DELLE PREVISIONI PER I PROSSIMI DUE GIORNI
        forecasts = []
        today = datetime.now().date()
        
        for item in data['list']:
            forecast_time = datetime.fromtimestamp(item['dt'])
            forecast_date = forecast_time.date()
            days_difference = (forecast_date - today).days
            if 0 <= days_difference <= 1:  # Solo oggi e domani
                forecasts.append({
                    'data': forecast_time,
                    'data_testo': forecast_time.strftime('%d/%m %H:%M'),
                    'giorno': days_difference,
                    'aqi': item['main']['aqi'],
                    'co': item['components']['co'],
                    'no2': item['components']['no2'],
                    'so2': item['components']['so2'],
                    'o3': item['components']['o3'],  # Aggiunto O3
                    'pm2_5': item['components']['pm2_5'],
                    'pm10': item['components']['pm10']
                })
        return {
            'forecasts': forecasts
        }
    except Exception as e:
        st.error(f"Errore durante il recupero previsioni sulla qualità dell'aria: {str(e)}")
        return None

#FUNZIONE CHE SALVA I DATI DELLA QUALITA' DELL'ARIA SU UN FILE CSV
def save_data_to_csv(data: dict, location: str, region: str, lat: float, lon: float) -> pd.DataFrame:
    """Salva i dati sulla qualità dell'aria in un file CSV e restituisce il DataFrame"""
    try:
        location = location.strip().title()
        region = normalize_region(region, location)
        components = data['list'][0]['components']
        aqi = data['list'][0]['main']['aqi']
        row = {
            "Regione": region,
            "Località": location,
            "Latitudine": lat,
            "Longitudine": lon,
            "Data": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "AQI": aqi,
            "CO": components.get("co", 0),
            "NO2": components.get("no2", 0),
            "SO2": components.get("so2", 0),
            "O3": components.get("o3", 0),  # Aggiunto O3
            "PM2.5": components.get("pm2_5", 0),
            "PM10": components.get("pm10", 0)
        }
        df = pd.DataFrame([row])
        filename = "air_quality_data.csv"
        
        # Crea il file o aggiungi la riga
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=False, index=False, quoting=1)
        else:
            df.to_csv(filename, index=False, quoting=1)
            
        logger.info(f"Dati salvati correttamente per {location}, {region}")
        return df
    except Exception as e:
        logger.error(f"Errore durante il salvataggio: {str(e)}")
        st.error(f"Errore durante il salvataggio: {str(e)}")
        return None

# FUNZIONE UNIFICATA PER L'ELABORAZIONE DELLE PREVISIONI
def process_forecast_data(forecast_data, forecast_type="weather"):
    """Elabora i dati di previsione in un formato leggibile"""
    summary = ""
    if not forecast_data or 'forecasts' not in forecast_data or not forecast_data['forecasts']:
        return summary
        
    #RAGGRUPPAMENTO PER GIORNI
    forecast_by_day = {}
    for fc in forecast_data['forecasts']:
        day = fc['giorno']
        if day not in forecast_by_day:
            forecast_by_day[day] = []
        forecast_by_day[day].append(fc)
        
    for day, forecasts in forecast_by_day.items():
        day_label = "Oggi" if day == 0 else "Domani"
        
        if forecast_type == "weather":
            avg_temp = sum(fc['temperatura'] for fc in forecasts) / len(forecasts)
            avg_wind = sum(fc['vento_velocita'] for fc in forecasts) / len(forecasts)
            rain_prob = max(fc['probabilita_pioggia'] for fc in forecasts)
            summary += f"""
            {day_label}: Temperatura media {avg_temp:.1f}°C, 
            Vento {avg_wind:.1f} m/s, 
            Probabilità pioggia {rain_prob:.0f}%
            """
        else:  
            avg_aqi = sum(fc['aqi'] for fc in forecasts) / len(forecasts)
            avg_pm25 = sum(fc['pm2_5'] for fc in forecasts) / len(forecasts)
            avg_pm10 = sum(fc['pm10'] for fc in forecasts) / len(forecasts)
            avg_o3 = sum(fc['o3'] for fc in forecasts) / len(forecasts)  # Aggiunto O3
            summary += f"""
            {day_label}: AQI medio {avg_aqi:.1f}, 
            PM2.5 {avg_pm25:.1f} μg/m³, 
            PM10 {avg_pm10:.1f} μg/m³,
            O3 {avg_o3:.1f} μg/m³
            """
            
    return summary

# FUNZIONI PER LA GESTIONE DEI DOCUMENTI
def pdf_extractor(file_path):
    """Estrae il testo da un file PDF"""
    try:
        logger.info(f"Tentativo di estrazione da {file_path}")
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        if not text.strip():
            logger.warning(f"PDF vuoto o non leggibile: {file_path}")
            return "Documento vuoto o non leggibile"
        logger.info(f"Estrazione completata: {len(text)} caratteri")
        return text
    except Exception as e:
        logger.error(f"Errore nell'estrazione PDF {file_path}: {str(e)}")
        return f"Errore nell'estrazione: {str(e)}"

# SISTEMA RAG
def load_rag_index():
    """Carica e indicizza i documenti per il sistema RAG usando ChromaDB"""
    try:
        if not Settings.embed_model:
            setup_embedding_model()   
        logger.info("Caricamento knowledge base...")
        # VERIFICA ESISTENZA CARTELLA 
        if not os.path.exists(KNOWLEDGE_BASE_DIR):
            os.makedirs(KNOWLEDGE_BASE_DIR)
            logger.warning(f"Cartella {KNOWLEDGE_BASE_DIR} creata")
        # VERIFICA ESISTENZA INDICE CHROMADB
        existing_index = load_chroma_index()
        if existing_index:
            return existing_index    
        # IF NOT EXIST VERIFICA SE CI SONO DOCUMENTI DA INDICIZZARE
        files = os.listdir(KNOWLEDGE_BASE_DIR)
        if not files:
            logger.warning("Knowledge base vuota, impossibile creare l'indice")
            # Crea un indice vuoto
            empty_index = create_chroma_index([Document(text="Indice iniziale vuoto.")])
            return empty_index   
        # CARICAMENTO DEI DOCUMENTI DALLA KNOWLEDGE BASE
        try:
            reader = SimpleDirectoryReader(
                input_dir=KNOWLEDGE_BASE_DIR,
                recursive=True,
                required_exts=[".pdf", ".txt"],
                file_metadata=lambda x: {"filename": os.path.basename(x), "type": "knowledge_base"},
                file_extractor={".pdf": pdf_extractor}
            )
            documents = reader.load_data()
        except Exception as e:
            logger.error(f"Errore nel caricamento documenti: {str(e)}")
            documents = []
        # IF NOT POSSIBLE TO CARICARE I DOCUMENTI, CREAZIONE INDICE VUOTO
        if not documents:
            logger.warning("Nessun documento valido trovato, creazione indice vuoto")
            empty_index = create_chroma_index([Document(text="Indice iniziale vuoto.")])
            return empty_index
        # CREAZIONE INDICE VUOTO CHROMA DB CON DOCUMENTI CARICATI
        logger.info(f"Creazione indice ChromaDB con {len(documents)} documenti...")
        return create_chroma_index(documents)
    except Exception as e:
        logger.error(f"Errore RAG: {str(e)}", exc_info=True)
        st.error(f"Errore nel sistema RAG: {str(e)}")
        return None
    
# FUNZIONE CHE RECUPERA I DOCUMENTI RILEVANTI DALL'INDICE VETTORIALE
def retrieve_rag_context(index, query, location=None, region=None, top_k=3):
    """Recupera documenti rilevanti dal database vettoriale"""
    try:
        if index is None:
            logger.warning("Indice RAG non disponibile")
            return ["Conoscenza base non disponibile"]
        
        # CONFIGURAZIONE RETRIEVER
        retriever = index.as_retriever(similarity_top_k=top_k)
        
        # APPLICAZIONE FILTRI SE SPECIFICATI
        if location or region:
            from llama_index.core.vector_stores import MetadataFilters, FilterOperator, MetadataFilter
            
            filters = []
            if location:
                filters.append(MetadataFilter(key="location", value=location, operator=FilterOperator.EQ))
            if region:
                filters.append(MetadataFilter(key="region", value=region, operator=FilterOperator.EQ))
                
            metadata_filters = MetadataFilters(filters=filters)
            
            retriever = index.as_retriever(
                similarity_top_k=top_k,
                filters=metadata_filters
            )
        
        # ESECUZIONE RETRIEVAL
        retrieval_results = retriever.retrieve(query)
        
        # ESTRAE E FORMATTA I RISULTATI
        context = []
        for node in retrieval_results:
            # FORMATTAZIONE DEI METADATI IN MODO LEGGIBILE
            metadata = node.metadata or {}
            source_type = metadata.get("type", "documento")
            
            prefix = ""
            if source_type == "air_quality_data":
                location_info = metadata.get("location", "")
                timestamp = metadata.get("timestamp", "")
                aqi = metadata.get("aqi", "")
                prefix = f"[Dati qualità aria: {location_info}, {timestamp}, AQI: {aqi}]\n"
            elif "filename" in metadata:
                prefix = f"[Fonte: {metadata.get('filename')}]\n"
            
            # AGGIUNTA DEL TESTO FORMATTATO
            context.append(f"{prefix}{node.text}")
        
        logger.info(f"Recuperati {len(context)} documenti rilevanti per la query: '{query}'")
        st.session_state.rag_context = context  # Salva per visualizzazione
        return context
    except Exception as e:
        logger.error(f"Errore nel retrieval documenti: {str(e)}")
        return [f"Errore nel recupero dei documenti: {str(e)}"]

# FUNZIONI PER L'INTERPRETAZIONE DELLA QUALITA' DELL'ARIA
def prepare_air_quality_context(components, weather_forecast=None, air_forecast=None):
    """Prepara il contesto per l'analisi della qualità dell'aria"""
    
    # PREPARA I DATI SUI COMPONENTI
    formatted_components = []
    for key, value in components.items():
        if key == "co":
            formatted_components.append(f"CO: {value} μg/m³")
        elif key == "no2":
            formatted_components.append(f"NO2: {value} μg/m³")
        elif key == "so2":
            formatted_components.append(f"SO2: {value} μg/m³")
        elif key == "o3":
            formatted_components.append(f"O3: {value} μg/m³")  # Aggiunto O3
        elif key == "pm2_5":
            formatted_components.append(f"PM2.5: {value} μg/m³")
        elif key == "pm10":
            formatted_components.append(f"PM10: {value} μg/m³")
    
    components_text = "\n".join(formatted_components)
    
    # DEFINIZIONE DEI LIMITI DI RIFERIMENTO PER OGNI INQUINANTE
    eu_limits = {
        "PM2.5": {"annual": 25, "who": 5},
        "PM10": {"annual": 40, "daily": 50, "who": 15},
        "NO2": {"annual": 40, "hourly": 200, "who": 10},
        "SO2": {"daily": 125, "hourly": 350, "who": 40},
        "O3": {"8hour": 120, "who": 100},  # Aggiunto O3
        "CO": {"8hour": 10000, "who": 4000}  # in μg/m³
    }
    
    # COSTRUZIONE MESSAGGI DI CONFRONTO 
    comparisons = []
    for pollutant, limits in eu_limits.items():
        key = pollutant.lower().replace(".", "_")
        value = components.get(key, 0)
        
        # Normalizza CO da mg/m³ a μg/m³ se necessario
        if key == "co" and value < 100:  # Presumibilmente già in mg/m³
            value = value * 1000
        
        eu_limit = limits.get("annual") or limits.get("daily") or limits.get("8hour")
        who_limit = limits.get("who")
        
        # VERIFICA VALORI 
        eu_status = "SOTTO" if value < eu_limit else "SOPRA"
        who_status = "SOTTO" if value < who_limit else "SOPRA"
        
        comparisons.append(f"{pollutant}: {value:.2f} μg/m³ - {eu_status} il limite UE ({eu_limit} μg/m³), {who_status} la raccomandazione OMS ({who_limit} μg/m³)")
    
    comparison_text = "\n".join(comparisons)
    
    # PREPARAZIONE INFORMAZIONI SUL METEO E SULLE PREVISIONI
    weather_text = process_forecast_data(weather_forecast, "weather") if weather_forecast else ""
    air_forecast_text = process_forecast_data(air_forecast, "air") if air_forecast else ""
    
    # UNIFICAZIONE DEL TESTO
    context = f"""
    ### VALORI ATTUALI INQUINANTI CON LIMITI DI RIFERIMENTO:
    {comparison_text}
    
    ### PREVISIONI METEO:
    {weather_text}
    
    ### PREVISIONI QUALITÀ ARIA:
    {air_forecast_text}
    """
    
    return context

# GESTIONE DEL MODELLO AI 
def call_ai_model(model_name, system_prompt, user_prompt):
    """Gestisce la chiamata al modello AI appropriato"""
    
    # CASO GITHUB/GPT4 
    if model_name == "github":
        try:
            if not GITHUB_TOKEN:
                return "⚠️ Token GitHub non configurato. Controlla il file .env"
            
            # CREAZIONE CLIENT OPENAI CON AZURE
            client = OpenAI(
                base_url=GITHUB_AZURE_ENDPOINT,
                api_key=GITHUB_TOKEN,
            )
            if not client:
                return "⚠️ Impossibile creare il client GitHub. Verifica il tuo token e l'endpoint."
            
            # EFFETTUAZIONE RICHIESTA ULITIZZANDO IL CLIENT OPENAI
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=GITHUB_MODEL_NAME,
                temperature=0.1,
                max_tokens=1000,
                top_p=1.0
            )
            # ESTRAZIONE DELLA RISPOSTA DAL MODELLO
            model_response = response.choices[0].message.content
            return model_response
            
        except Exception as e:
            logger.error(f"Errore durante la chiamata a GitHub/OpenAI: {str(e)}")
            return f"⚠️ Errore durante la connessione a GitHub: {str(e)}. Verifica il tuo token."
    
    # CASE COHERE
    elif model_name == "cohere":
        try:
            if not COHERE_API_KEY:
                return "⚠️ API key Cohere non configurata. Controlla il file .env"
            
            # PREPARAZIONE RICHIESTA
            headers = {
                "Authorization": f"Bearer {COHERE_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Aggiungi questo all'inizio del system_prompt
            italian_instruction = "IMPORTANTE: RISPONDI SEMPRE E SOLO IN ITALIANO. "
            enhanced_system_prompt = italian_instruction + system_prompt
            
            payload = {
                "message": user_prompt,
                "model": "command",
                "temperature": 0.1,
                "max_tokens": 1000,
                "preamble": enhanced_system_prompt,
                "language": "italian" 
            }
            
            # EFFETTUAZIONE RICHIESTA
            response = requests.post(COHERE_API_ENDPOINT, json=payload, headers=headers, timeout=300)
            response.raise_for_status()
            
            # ESTRAZIONE RISPOSTA DAL JSON 
            model_response = response.json().get('text', 'Nessuna risposta generata')
            return model_response
                
        except Exception as e:
            logger.error(f"Errore durante la chiamata a Cohere: {str(e)}")
            return f"⚠️ Errore durante la connessione a Cohere: {str(e)}. Verifica la tua API key."
    
    # Fallback su GitHub se il modello non è riconosciuto
    else:
        logger.warning(f"Modello {model_name} non riconosciuto, utilizzo di GitHub come fallback")
        return call_ai_model("github", system_prompt, user_prompt)
    
# FUNZIONE CHE INTERPRETA LA QUALITA' DELL'ARIA 
def interpret_air_quality(components, model_name, question, weather_forecast=None, air_forecast=None, rag_index=None):
    """Versione ottimizzata della funzione per interpretare la qualità dell'aria con RAG"""
    try:
        
        if not question or not question.strip():
            return {"llm": "Per favore, inserisci una domanda valida.", "rag": "Per favore, inserisci una domanda valida."}
        
        
        context = prepare_air_quality_context(components, weather_forecast, air_forecast)
        
        # PROMPT SISTEMA CON LIMITAZIONE LUNGHEZZA
        system_prompt = """
        Agisci come esperto di qualità dell'aria. Spiega e dammi la risposta in italiano.
        Analizza con precisione i dati di inquinamento forniti nel CONTESTO.
        Rispondi in modo chiaro, preciso e conciso (massimo 25 righe, senza elenchi numerati), 
        andando dritto al punto con frasi brevi. Confronta la qualità dell'aria con i limiti attuali fissati 
        sia dall’Unione Europea che dall’Organizzazione Mondiale della Sanità, citando i valori soglia ufficiali 
        in vigore nel 2023 o aggiornati successivamente. Se possibile, indica in che modo i valori rilevati 
        si discostano dalle soglie ufficiali, specificando se si tratta di superamenti giornalieri, 
        annuali o orari, e spiegando quali rischi sanitari ne derivano secondo le fonti ufficiali.
        Includi l’analisi dell’ozono (O₃) solo se rilevante, e riporta anche qui i limiti normativi precisi e aggiornati.
        Evita generalizzazioni: ogni affermazione deve essere supportata da fonti, 
        dati ufficiali o normative documentate. Se mancano i dati nel contesto, segnala chiaramente l'incertezza. 
        """
        
        # PROMPT RISPOSTA BASE LLM
        user_prompt_base = f"""
        ### DOMANDA:
        {question}
        ### CONTESTO:
        {context}
        """
        # CHIAMATA MODELLO
        llm_response = call_ai_model(model_name, system_prompt, user_prompt_base)
        
        # PREPARAZIONE RISPOSTA RAG+LLM
        rag_response = "RAG non disponibile"
        
        if rag_index:
            
            location = st.session_state.get('location', None)
            region = st.session_state.get('region', None)
            
            rag_context = retrieve_rag_context(
                rag_index, 
                question, 
                location=location,
                region=region,
                top_k=10
            )
            
            rag_prompt = user_prompt_base
            if rag_context:
                rag_prompt += "\n\n### CONOSCENZA AGGIUNTIVA:\n"
                for ctx in rag_context:
                    rag_prompt += f"{ctx}\n\n"
            
            rag_response = call_ai_model(model_name, system_prompt, rag_prompt)
        
        def clean_response(response):
            if "Conclusione" in response:
                parts = response.split("Conclusione")
                response = parts[0].strip()
            return response
        
        llm_response = clean_response(llm_response)
        rag_response = clean_response(rag_response)
        
        # COMANDO PER OTTENERE MASSIMO 25 RIGHE DI RISPOSTA PER CIASCUNA RISPOSTA
        def trim_response(response, max_lines=25):
            lines = response.strip().split('\n')
            if len(lines) > max_lines:
                return '\n'.join(lines[:max_lines]) + "\n(Risposta limitata per brevità)"
            return response
        
        llm_response = trim_response(llm_response)
        rag_response = trim_response(rag_response)
        
        # VERIFICA QUALITA' RISPOSTA
        if len(llm_response.strip()) < 50 or llm_response.count(".") < 3:
            llm_response = f"""
            Mi dispiace, non sono riuscito a generare una risposta adeguata alla tua domanda.
            
            Ti suggerisco di provare a riformulare la domanda in modo più specifico o di selezionare una delle domande predefinite.
            """
        
        return {"llm": llm_response, "rag": rag_response}

    except Exception as e:
        logger.error(f"Errore durante l'analisi: {str(e)}")
        return {"llm": f"❌ Errore durante l'analisi: {str(e)}", "rag": f"❌ Errore durante l'analisi: {str(e)}"}
    
# DEFINIZIONE DELLE DIMENSIONI ANALITICHE
class AnalysisDimension(Enum):
    ACCURACY = "accuratezza_veridicita"
    COMPLETENESS = "completezza_informativa"
    HALLUCINATION_ROBUSTNESS = "robustezza_allucinazioni"
    LINGUISTIC_QUALITY = "qualita_linguistica"
    PERCEIVED_UTILITY = "utilita_percepita"
    READABILITY = "accessibilita_leggibilita"
    SOURCE_LINKING = "collegamento_fonti"  # Nuova dimensione
    TEMPORAL_VALIDITY = "validita_temporale"  # Nuova dimensione

# Mapping delle metriche dettagliate alle dimensioni
metric_to_dimension = {
    # ACCURACY
    "factual_consistency": AnalysisDimension.ACCURACY,
    "justification_alignment": AnalysisDimension.ACCURACY,
    "explicit_source_accuracy": AnalysisDimension.ACCURACY,
    "numerical_integrity_check": AnalysisDimension.ACCURACY,
    
    # COMPLETENESS
    "coverage_score": AnalysisDimension.COMPLETENESS,
    "content_density": AnalysisDimension.COMPLETENESS,
    "specificity_score": AnalysisDimension.COMPLETENESS,
    
    # HALLUCINATION_ROBUSTNESS
    "hallucination_rate": AnalysisDimension.HALLUCINATION_ROBUSTNESS,
    "source_backed_claim_ratio": AnalysisDimension.HALLUCINATION_ROBUSTNESS,
    "uncertainty_reporting": AnalysisDimension.HALLUCINATION_ROBUSTNESS,
    
    # LINGUISTIC_QUALITY
    "fluency_score": AnalysisDimension.LINGUISTIC_QUALITY,
    "coherence_score": AnalysisDimension.LINGUISTIC_QUALITY,
    
    # PERCEIVED_UTILITY
    "user_utility": AnalysisDimension.PERCEIVED_UTILITY,
    "interrogability_score": AnalysisDimension.PERCEIVED_UTILITY,
    
    # READABILITY
    "readability_score": AnalysisDimension.READABILITY,
    "technical_jargon_count": AnalysisDimension.READABILITY,
    "sentence_length_avg": AnalysisDimension.READABILITY,
    
    # SOURCE_LINKING
    "evidence_linking_score": AnalysisDimension.SOURCE_LINKING,
    "answer_traceability": AnalysisDimension.SOURCE_LINKING,
    
    # TEMPORAL_VALIDITY
    "temporal_validity": AnalysisDimension.TEMPORAL_VALIDITY
}

@dataclass
class ResponseAnalysis:
    """Rappresenta l'analisi di una singola risposta"""
    source: str  # "LLM" o "RAG+LLM"
    metrics: Dict[str, float]  # Score per ogni metrica
    dimension_scores: Dict[AnalysisDimension, float]  # Score aggregati per dimensione
    flags: List[str]  # Problemi rilevati
    word_count: int
    sentence_count: int
    specific_facts: List[str]
    potential_hallucinations: List[str]
    confidence_level: float

def calculate_dimension_scores(metrics: Dict[str, float]) -> Dict[AnalysisDimension, float]:
    """Calcola i punteggi per le otto dimensioni di analisi a partire dalle metriche dettagliate."""
    dimension_scores = {}
    
    # Inizializza i punteggi delle dimensioni
    for dimension in AnalysisDimension:
        dimension_scores[dimension] = 0.0
    
    # Raggruppa le metriche per dimensione e calcola la media
    dimension_metrics_count = {dimension: 0 for dimension in AnalysisDimension}
    
    for metric, value in metrics.items():
        if metric in metric_to_dimension:
            dimension = metric_to_dimension[metric]
            
            # Gestione speciale per i conteggi (normalizzazione a scala 0-1)
            if metric == "technical_jargon_count":
                # Più gergo = peggio (scala 0-20, invertita)
                normalized_value = max(0, min(1, 1 - (value / 20)))
                dimension_scores[dimension] += normalized_value
            elif metric == "sentence_length_avg":
                # Lunghezza ottimale delle frasi intorno a 15-20 parole
                # Troppo corto o troppo lungo = peggio
                normalized_value = max(0, min(1, 1 - abs(value - 17.5) / 15))
                dimension_scores[dimension] += normalized_value
            elif metric == "hallucination_rate":
                # Questo è già invertito (più alto = meno allucinazioni)
                dimension_scores[dimension] += value / 100
            else:
                # Metriche percentuali regolari
                dimension_scores[dimension] += value / 100
            
            dimension_metrics_count[dimension] += 1
    
    # Calcola la media per ogni dimensione
    for dimension in dimension_scores:
        if dimension_metrics_count[dimension] > 0:
            dimension_scores[dimension] /= dimension_metrics_count[dimension]
    
    return dimension_scores

def extract_specific_facts(text: str) -> List[str]:
    """Estrae fatti specifici dal testo (date, numeri, nomi propri)"""
    facts = []
    
    # Pattern per date
    date_pattern = r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b|\b\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}\b'
    dates = re.findall(date_pattern, text)
    facts.extend([f"Data: {date}" for date in dates])
    
    # Pattern per numeri specifici
    number_pattern = r'\b\d+\.?\d*\s*(?:€|%|km|metri|kg|anni|volte|μg|mg|m³)\b'
    numbers = re.findall(number_pattern, text, re.IGNORECASE)
    facts.extend([f"Numero: {num}" for num in numbers])
    
    # Pattern per nomi propri (maiuscole consecutive)
    name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    names = re.findall(name_pattern, text)
    facts.extend([f"Nome: {name}" for name in names if len(name) > 3])
    
    return facts

def analyze_single_response(response: str, source: str, model_name: str) -> ResponseAnalysis:
    """Analizza una singola risposta per vari parametri di qualità"""
    
    # Analisi quantitativa di base
    word_count = len(response.split())
    sentence_count = len(re.split(r'[.!?]+', response))
    
    # Estrazione fatti specifici (date, numeri, nomi propri)
    specific_facts = extract_specific_facts(response)
    
    # Sistema di prompt per analisi dettagliata
    analysis_prompt = f"""
    Analizza la seguente risposta secondo questi criteri dettagliati, valutando su una scala da 0 a 100:

    1. ACCURATEZZA E VERIDICITÀ
       - factual_consistency (0-100): Quanto è fattualmente corretta la risposta rispetto alla documentazione di riferimento?
       - justification_alignment (0-100): Quanto le argomentazioni sono logicamente connesse con le conclusioni?
       - explicit_source_accuracy (0-100): Quanto sono accurate e veritiere le citazioni esplicite delle fonti?
       - numerical_integrity_check (0-100): Quanto sono accurati i dati numerici e le statistiche presentate?

    2. COMPLETEZZA INFORMATIVA
       - coverage_score (0-100): Quanto copre i concetti chiave rispetto a una risposta ideale?
       - content_density (0-100): Qual è la densità informativa, ovvero il rapporto tra contenuti utili e lunghezza?
       - specificity_score (0-100): Quanto è specifica e dettagliata la risposta sui punti chiave?

    3. ROBUSTEZZA ALLE ALLUCINAZIONI
       - hallucination_rate (0-100): Qual è la frequenza di affermazioni non supportate da evidenze? (più alto = meno allucinazioni)
       - source_backed_claim_ratio (0-100): Percentuale di affermazioni supportate da fonti verificabili.
       - uncertainty_reporting (0-100): Quanto efficacemente vengono segnalate le incertezze o la mancanza di dati?

    4. QUALITÀ LINGUISTICA
       - fluency_score (0-100): Quanto è grammaticalmente corretta e fluida la risposta?
       - coherence_score (0-100): Quanto è coerente logicamente e strutturalmente il contenuto?

    5. UTILITÀ PERCEPITA
       - user_utility (0-100): Qual è l'utilità soggettiva della risposta per un utente?
       - interrogability_score (0-100): Quanto la risposta stimola ulteriori domande o approfondimenti pertinenti?

    6. ACCESSIBILITÀ E LEGGIBILITÀ
       - readability_score (0-100): Quanto è leggibile il testo (più alto = più leggibile)?
       - technical_jargon_count (numero intero): Quanti termini tecnici non contestualizzati o spiegati sono presenti?
       - sentence_length_avg (numero float): Qual è la lunghezza media delle frasi in parole?

    7. COLLEGAMENTO FONTI
       - evidence_linking_score (0-100): Quanto efficacemente la risposta collega le affermazioni alle evidenze?
       - answer_traceability (0-100): Quanto facilmente si possono tracciare le fonti di ciascuna affermazione?

    8. VALIDITÀ TEMPORALE
       - temporal_validity (0-100): Quanto la risposta considera appropriatamente la dimensione temporale delle informazioni?

    Rispondi SOLO in formato JSON:
    {{
        "factual_consistency": [score],
        "justification_alignment": [score],
        "explicit_source_accuracy": [score],
        "numerical_integrity_check": [score],
        "coverage_score": [score],
        "content_density": [score],
        "specificity_score": [score],
        "hallucination_rate": [score],
        "source_backed_claim_ratio": [score],
        "uncertainty_reporting": [score],
        "fluency_score": [score],
        "coherence_score": [score],
        "user_utility": [score],
        "interrogability_score": [score],
        "readability_score": [score],
        "technical_jargon_count": [numero],
        "sentence_length_avg": [numero],
        "evidence_linking_score": [score],
        "answer_traceability": [score],
        "temporal_validity": [score],
        "problemi_rilevati": ["problema1", "problema2"],
        "potenziali_allucinazioni": ["allucinazione1", "allucinazione2"],
        "livello_fiducia": [score 0-100]
    }}

    RISPOSTA DA ANALIZZARE:
    {response}
    """
    
    try:
        # Chiamata all'AI per analisi dettagliata
        analysis_result = call_ai_model(model_name, "Sei un esperto analista di testi.", analysis_prompt)
        analysis_data = json.loads(analysis_result)
        
        # Estrai metriche dall'analisi
        metrics = {}
        for key, value in analysis_data.items():
            if key not in ["problemi_rilevati", "potenziali_allucinazioni", "livello_fiducia"]:
                metrics[key] = value
        
        # Calcola i punteggi per dimensione
        dimension_scores = calculate_dimension_scores(metrics)
        
        return ResponseAnalysis(
            source=source,
            metrics=metrics,
            dimension_scores=dimension_scores,
            flags=analysis_data.get("problemi_rilevati", []),
            word_count=word_count,
            sentence_count=sentence_count,
            specific_facts=specific_facts,
            potential_hallucinations=analysis_data.get("potenziali_allucinazioni", []),
            confidence_level=analysis_data.get("livello_fiducia", 0) / 100.0
        )
        
    except Exception as e:
        logger.error(f"Errore nell'analisi della risposta {source}: {str(e)}")
        # Fallback con analisi semplificata
        return create_fallback_analysis(response, source)

def create_fallback_analysis(response: str, source: str) -> ResponseAnalysis:
    """Crea un'analisi semplificata in caso di errore nel processo principale"""
    word_count = len(response.split())
    sentence_count = len(re.split(r'[.!?]+', response))
    
    # Metriche semplici basate su euristiche
    metrics = {
        # ACCURACY
        "factual_consistency": 70,
        "justification_alignment": 70,
        "explicit_source_accuracy": 70,
        "numerical_integrity_check": 70,
        
        # COMPLETENESS
        "coverage_score": min(100, word_count / 2),
        "content_density": 70,
        "specificity_score": 70,
        
        # HALLUCINATION_ROBUSTNESS
        "hallucination_rate": 70,  # Più alto = meno allucinazioni
        "source_backed_claim_ratio": 70,
        "uncertainty_reporting": 70,
        
        # LINGUISTIC_QUALITY
        "fluency_score": 80,
        "coherence_score": 75,
        
        # PERCEIVED_UTILITY
        "user_utility": 70,
        "interrogability_score": 70,
        
        # READABILITY
        "readability_score": 75,
        "technical_jargon_count": 5,
        "sentence_length_avg": 20,
        
        # SOURCE_LINKING
        "evidence_linking_score": 70,
        "answer_traceability": 70,
        
        # TEMPORAL_VALIDITY
        "temporal_validity": 70
    }
    
    # Calcola i punteggi per dimensione
    dimension_scores = calculate_dimension_scores(metrics)
    
    return ResponseAnalysis(
        source=source,
        metrics=metrics,
        dimension_scores=dimension_scores,
        flags=["Analisi semplificata utilizzata"],
        word_count=word_count,
        sentence_count=sentence_count,
        specific_facts=[],
        potential_hallucinations=[],
        confidence_level=0.5
    )

def perform_cross_verification(llm_analysis: ResponseAnalysis, rag_analysis: ResponseAnalysis, model_name: str) -> Dict:
    """Verifica incrociata tra le due risposte per identificare incongruenze"""
    
    verification_prompt = f"""
    Confronta queste due analisi di risposte e identifica:
    1. Fatti che solo una delle due include
    2. Possibili contraddizioni tra le risposte
    3. Livello di dettaglio comparativo
    
    Rispondi in formato JSON:
    {{
        "fatti_unici_llm": ["fatto1", "fatto2"],
        "fatti_unici_rag": ["fatto1", "fatto2"],
        "contraddizioni": ["contraddizione1"],
        "livello_dettaglio_comparativo": "LLM/RAG/PARI"
    }}
    
    FATTI RISPOSTA LLM:
    {llm_analysis.specific_facts}
    
    FATTI RISPOSTA RAG:
    {rag_analysis.specific_facts}
    """
    
    try:
        verification_result = call_ai_model(model_name, "Sei un esperto verificatore di fatti.", verification_prompt)
        return json.loads(verification_result)
    except Exception as e:
        logger.error(f"Errore nella verifica incrociata: {str(e)}")
        return {}

def calculate_weighted_score(analysis: ResponseAnalysis) -> float:
    """Calcola un punteggio pesato per l'analisi"""
    # Pesi per ogni dimensione
    dimension_weights = {
        AnalysisDimension.ACCURACY: 0.20,
        AnalysisDimension.COMPLETENESS: 0.15,
        AnalysisDimension.HALLUCINATION_ROBUSTNESS: 0.20,
        AnalysisDimension.LINGUISTIC_QUALITY: 0.10,
        AnalysisDimension.PERCEIVED_UTILITY: 0.10,
        AnalysisDimension.READABILITY: 0.05,
        AnalysisDimension.SOURCE_LINKING: 0.15,
        AnalysisDimension.TEMPORAL_VALIDITY: 0.05
    }
    
    total_score = 0.0
    for dimension, weight in dimension_weights.items():
        if dimension in analysis.dimension_scores:
            total_score += weight * analysis.dimension_scores[dimension]
    
    # Normalizzazione tra 0 e 1
    return max(0.0, min(1.0, total_score))

def create_comparison_visualization(llm_analysis, rag_analysis, llm_score, rag_score):
    """
    Crea una visualizzazione grafica del confronto tra l'analisi LLM e RAG+LLM
    utilizzando un radar chart unificato per le otto dimensioni.
    
    Args:
        llm_analysis: ResponseAnalysis per la risposta LLM
        rag_analysis: ResponseAnalysis per la risposta RAG+LLM
        llm_score: Punteggio pesato finale per LLM
        rag_score: Punteggio pesato finale per RAG+LLM
        
    Returns:
        fig: Figura Plotly da visualizzare
    """
    # Crea una figura
    fig = go.Figure()
    
    # Definisce l'ordine e le etichette per le dimensioni
    dimensions = [
        AnalysisDimension.ACCURACY,
        AnalysisDimension.COMPLETENESS,
        AnalysisDimension.HALLUCINATION_ROBUSTNESS,
        AnalysisDimension.LINGUISTIC_QUALITY,
        AnalysisDimension.PERCEIVED_UTILITY,
        AnalysisDimension.READABILITY,
        AnalysisDimension.SOURCE_LINKING,
        AnalysisDimension.TEMPORAL_VALIDITY
    ]
    
    # Traduzione per la visualizzazione
    dimension_labels = {
        AnalysisDimension.ACCURACY: "Accuratezza e\nVeridicità",
        AnalysisDimension.COMPLETENESS: "Completezza\nInformativa",
        AnalysisDimension.HALLUCINATION_ROBUSTNESS: "Robustezza alle\nAllucinazioni",
        AnalysisDimension.LINGUISTIC_QUALITY: "Qualità\nLinguistica",
        AnalysisDimension.PERCEIVED_UTILITY: "Utilità\nPercepita",
        AnalysisDimension.READABILITY: "Accessibilità e\nLeggibilità",
        AnalysisDimension.SOURCE_LINKING: "Collegamento\nFonti",
        AnalysisDimension.TEMPORAL_VALIDITY: "Validità\nTemporale"
    }
    
    # Estrae i punteggi delle dimensioni
    llm_values = [llm_analysis.dimension_scores[dim] for dim in dimensions]
    rag_values = [rag_analysis.dimension_scores[dim] for dim in dimensions]
    labels = [dimension_labels[dim] for dim in dimensions]
    
    # Chiude il ciclo per il radar chart
    llm_values.append(llm_values[0])
    rag_values.append(rag_values[0])
    labels.append(labels[0])
    
    # Aggiunge traccia LLM
    fig.add_trace(
        go.Scatterpolar(
            r=llm_values,
            theta=labels,
            fill='toself',
            name='LLM',
            line_color='rgba(31, 119, 180, 0.8)',
            fillcolor='rgba(31, 119, 180, 0.3)'
        )
    )
    
    # Aggiunge traccia RAG+LLM
    fig.add_trace(
        go.Scatterpolar(
            r=rag_values,
            theta=labels,
            fill='toself',
            name='RAG+LLM',
            line_color='rgba(255, 127, 14, 0.8)',
            fillcolor='rgba(255, 127, 14, 0.3)'
        )
    )
    
    # Aggiunge annotazione per i punteggi finali
    winner = "LLM" if llm_score > rag_score else "RAG+LLM"
    fig.add_annotation(
        x=0.5, y=1.15,
        xref="paper", yref="paper",
        text=f"Punteggio finale: LLM = {llm_score:.3f} | RAG+LLM = {rag_score:.3f} | Vincitore: {winner}",
        showarrow=False,
        font=dict(
            family="Arial",
            size=14,
            color="#333"
        ),
        align="center",
    )
    
    # Aggiorna layout
    fig.update_layout(
        height=600,
        width=800,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10),
                tickvals=[0.2, 0.4, 0.6, 0.8],
                ticktext=["20%", "40%", "60%", "80%"]
            )
        ),
        title=dict(
            text="Analisi comparativa delle risposte",
            x=0.5,
            y=0.95
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=80, b=100)
    )
    
    return fig

def create_comparison_table(llm_analysis, rag_analysis):
    """
    Crea una tabella di confronto con i punteggi normalizzati per ogni dimensione.
    
    Args:
        llm_analysis: ResponseAnalysis per la risposta LLM
        rag_analysis: ResponseAnalysis per la risposta RAG+LLM
        
    Returns:
        fig: Figura Plotly con tabella di confronto
    """
    # Definisce l'ordine e le etichette per le dimensioni
    dimensions = [
        AnalysisDimension.ACCURACY,
        AnalysisDimension.COMPLETENESS,
        AnalysisDimension.HALLUCINATION_ROBUSTNESS,
        AnalysisDimension.LINGUISTIC_QUALITY,
        AnalysisDimension.PERCEIVED_UTILITY,
        AnalysisDimension.READABILITY,
        AnalysisDimension.SOURCE_LINKING,
        AnalysisDimension.TEMPORAL_VALIDITY
    ]
    
    # Traduzione per la visualizzazione
    dimension_labels = {
        AnalysisDimension.ACCURACY: "Accuratezza e Veridicità",
        AnalysisDimension.COMPLETENESS: "Completezza Informativa",
        AnalysisDimension.HALLUCINATION_ROBUSTNESS: "Robustezza alle Allucinazioni",
        AnalysisDimension.LINGUISTIC_QUALITY: "Qualità Linguistica",
        AnalysisDimension.PERCEIVED_UTILITY: "Utilità Percepita",
        AnalysisDimension.READABILITY: "Accessibilità e Leggibilità",
        AnalysisDimension.SOURCE_LINKING: "Collegamento Fonti",
        AnalysisDimension.TEMPORAL_VALIDITY: "Validità Temporale"
    }
    
    # Crea una tabella con i punteggi normalizzati
    headers = ["Dimensione", "LLM", "RAG+LLM", "Differenza"]
    cells = []
    
    for dim in dimensions:
        llm_score = llm_analysis.dimension_scores[dim]
        rag_score = rag_analysis.dimension_scores[dim]
        diff = rag_score - llm_score
        
        # Formatta i valori come percentuali
        cells.append([
            dimension_labels[dim],
            f"{llm_score*100:.3f}%",
            f"{rag_score*100:.3f}%",
            f"{diff*100:+.3f}%"  # Il segno + mostra esplicitamente se è positivo
        ])
    
    # Crea la figura della tabella
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color='#F5F5F5',
            align='center',
            font=dict(size=14)
        ),
        cells=dict(
            values=list(zip(*cells)),  # Trasponi cells per adattarla al formato values di Plotly
            fill_color=[
                ['white'] * len(cells),  # Colonna Dimensione
                ['rgba(31, 119, 180, 0.1)'] * len(cells),  # Colonna LLM
                ['rgba(255, 127, 14, 0.1)'] * len(cells),  # Colonna RAG+LLM
                [
                    'rgba(0, 255, 0, 0.1)' if float(cell[3][:-1]) > 0 else 
                    'rgba(255, 0, 0, 0.1)' if float(cell[3][:-1]) < 0 else 
                    'white' 
                    for cell in cells
                ]  # Colonna Differenza - verde se positivo, rosso se negativo
            ],
            align='center',
            font=dict(size=12)
        )
    )])
    
    fig.update_layout(
        height=350,  # Aumentato per ospitare più righe
        width=800,
        margin=dict(l=20, r=20, t=50, b=20),
        title="Confronto dettagliato dei punteggi per dimensione"
    )
    
    return fig

def save_comparison_result(question, decision, model_name, location, region):
    """
    Salva il risultato della comparazione tra LLM e RAG+LLM in un file CSV.
    
    Args:
        question (str): La domanda posta dall'utente
        decision (str): Il risultato della comparazione ("LLM" o "RAG+LLM" o "NESSUNA DIFFERENZA")
        model_name (str): Il nome del modello utilizzato
        location (str): La località analizzata
        region (str): La regione analizzata
    """
    try:
        import pandas as pd
        from datetime import datetime
        import os
        
        # Prepara i dati per il CSV
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Estrai solo il tipo di sistema dalla decisione (LLM, RAG+LLM o NESSUNA DIFFERENZA)
        if "NESSUNA DIFFERENZA" in decision:
            winner = "NESSUNA DIFFERENZA"
        elif "LLM" in decision and "RAG+LLM" not in decision:
            winner = "LLM"
        else:
            winner = "RAG+LLM"
        
        # Limita la lunghezza della domanda per evitare problemi nel CSV
        short_question = question[:100] + "..." if len(question) > 100 else question
        
        data = {
            "Timestamp": [timestamp],
            "Modello": [model_name],
            "Località": [location],
            "Regione": [region],
            "Domanda": [short_question],
            "Vincitore": [winner]
        }
        
        df = pd.DataFrame(data)
        
        # Definisci il percorso del file
        filename = "comparison_results.csv"
        
        # Controlla se il file esiste già
        if os.path.exists(filename):
            # Aggiungi la riga senza ripetere l'intestazione
            df.to_csv(filename, mode='a', header=False, index=False, quoting=1, encoding='utf-8')
        else:
            # Crea un nuovo file con l'intestazione
            df.to_csv(filename, index=False, quoting=1, encoding='utf-8')
            
        logger.info(f"Risultato della comparazione salvato con successo: {winner}")
        return True
    except Exception as e:
        logger.error(f"Errore durante il salvataggio del risultato della comparazione: {str(e)}")
        return False

def load_comparison_data():
    """
    Carica i dati delle comparazioni dal file CSV e li restituisce come DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame con i dati delle comparazioni, o None in caso di errore
    """
    try:
        import pandas as pd
        import os
        
        filename = "comparison_results.csv"
        
        if not os.path.exists(filename):
            return None
            
        df = pd.read_csv(filename, encoding='utf-8')
        
        # Converti la colonna del timestamp in datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        return df
    except Exception as e:
        logger.error(f"Errore durante il caricamento dei dati delle comparazioni: {str(e)}")
        return None

def analyze_comparison_data(df):
    """
    Analizza i dati delle comparazioni e restituisce statistiche rilevanti.
    
    Args:
        df (pd.DataFrame): DataFrame con i dati delle comparazioni
        
    Returns:
        dict: Dizionario con le statistiche calcolate
    """
    try:
        import pandas as pd
        
        if df is None or len(df) == 0:
            return {
                "total_comparisons": 0,
                "llm_wins": 0,
                "rag_wins": 0,
                "ties": 0,
                "llm_percentage": 0,
                "rag_percentage": 0,
                "tie_percentage": 0,
                "by_model": {},
                "by_day": {},
                "recent_trend": []
            }
        
        # Statistiche generali
        total = len(df)
        llm_wins = len(df[df['Vincitore'] == 'LLM'])
        rag_wins = len(df[df['Vincitore'] == 'RAG+LLM'])
        ties = len(df[df['Vincitore'] == 'NESSUNA DIFFERENZA'])
        
        llm_percentage = (llm_wins / total) * 100 if total > 0 else 0
        rag_percentage = (rag_wins / total) * 100 if total > 0 else 0
        tie_percentage = (ties / total) * 100 if total > 0 else 0
        
        # Statistiche per modello
        by_model = {}
        for model in df['Modello'].unique():
            model_df = df[df['Modello'] == model]
            model_total = len(model_df)
            model_llm_wins = len(model_df[model_df['Vincitore'] == 'LLM'])
            model_rag_wins = len(model_df[model_df['Vincitore'] == 'RAG+LLM'])
            model_ties = len(model_df[model_df['Vincitore'] == 'NESSUNA DIFFERENZA'])
            
            by_model[model] = {
                "total": model_total,
                "llm_wins": model_llm_wins,
                "rag_wins": model_rag_wins,
                "ties": model_ties,
                "llm_percentage": (model_llm_wins / model_total) * 100 if model_total > 0 else 0,
                "rag_percentage": (model_rag_wins / model_total) * 100 if model_total > 0 else 0,
                "tie_percentage": (model_ties / model_total) * 100 if model_total > 0 else 0
            }
        
        # Analisi temporale per giorno
        df['Date'] = df['Timestamp'].dt.date
        by_day = {}
        
        for date in df['Date'].unique():
            date_df = df[df['Date'] == date]
            date_total = len(date_df)
            date_llm_wins = len(date_df[date_df['Vincitore'] == 'LLM'])
            date_rag_wins = len(date_df[date_df['Vincitore'] == 'RAG+LLM'])
            date_ties = len(date_df[date_df['Vincitore'] == 'NESSUNA DIFFERENZA'])
            
            by_day[str(date)] = {
                "total": date_total,
                "llm_wins": date_llm_wins,
                "rag_wins": date_rag_wins,
                "ties": date_ties,
                "llm_percentage": (date_llm_wins / date_total) * 100 if date_total > 0 else 0,
                "rag_percentage": (date_rag_wins / date_total) * 100 if date_total > 0 else 0,
                "tie_percentage": (date_ties / date_total) * 100 if date_total > 0 else 0
            }
        
        # Trend recente (ultimi 10 risultati)
        recent = df.sort_values('Timestamp', ascending=False).head(10)
        recent_trend = recent['Vincitore'].tolist()
        
        return {
            "total_comparisons": total,
            "llm_wins": llm_wins,
            "rag_wins": rag_wins,
            "ties": ties,
            "llm_percentage": llm_percentage,
            "rag_percentage": rag_percentage,
            "tie_percentage": tie_percentage,
            "by_model": by_model,
            "by_day": by_day,
            "recent_trend": recent_trend
        }
    except Exception as e:
        logger.error(f"Errore durante l'analisi dei dati delle comparazioni: {str(e)}")
        return None

def create_comparison_statistics_charts(df):
    """
    Crea grafici per visualizzare le statistiche delle comparazioni.
    
    Args:
        df (pd.DataFrame): DataFrame con i dati delle comparazioni
        
    Returns:
        dict: Dizionario con i grafici generati
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
        
        if df is None or len(df) == 0:
            # Ritorna un grafico vuoto con un messaggio
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                text="Nessun dato disponibile",
                showarrow=False,
                font=dict(size=20)
            )
            return {"main_fig": fig}
        
        # 1. Grafico a torta per la distribuzione generale
        wins_count = df['Vincitore'].value_counts().reset_index()
        wins_count.columns = ['Sistema', 'Conteggio']
        
        pie_fig = px.pie(
            wins_count, 
            values='Conteggio', 
            names='Sistema',
            title="Distribuzione delle vittorie",
            color='Sistema',
            color_discrete_map={
                'LLM': 'rgba(31, 119, 180, 0.8)', 
                'RAG+LLM': 'rgba(255, 127, 14, 0.8)',
                'NESSUNA DIFFERENZA': 'rgba(44, 160, 44, 0.8)'
            },
            hole=0.4
        )
        
        pie_fig.update_traces(
            textinfo='percent+label+value',
            pull=[0.05, 0.05, 0.05],
            marker=dict(line=dict(color='#FFFFFF', width=2))
        )
        
        # 2. Analisi temporale
        df['Date'] = pd.to_datetime(df['Timestamp']).dt.date
        
        # Raggruppa per giorno e sistema
        daily_counts = df.groupby(['Date', 'Vincitore']).size().unstack(fill_value=0).reset_index()
        
        if 'LLM' not in daily_counts.columns:
            daily_counts['LLM'] = 0
        if 'RAG+LLM' not in daily_counts.columns:
            daily_counts['RAG+LLM'] = 0
        if 'NESSUNA DIFFERENZA' not in daily_counts.columns:
            daily_counts['NESSUNA DIFFERENZA'] = 0
        
        # Percentuali giornaliere
        daily_counts['Total'] = daily_counts['LLM'] + daily_counts['RAG+LLM'] + daily_counts['NESSUNA DIFFERENZA']
        daily_counts['LLM_PCT'] = (daily_counts['LLM'] / daily_counts['Total']) * 100
        daily_counts['RAG_PCT'] = (daily_counts['RAG+LLM'] / daily_counts['Total']) * 100
        daily_counts['TIE_PCT'] = (daily_counts['NESSUNA DIFFERENZA'] / daily_counts['Total']) * 100
        
        # Figura per i conteggi giornalieri
        time_fig = go.Figure()
        
        time_fig.add_trace(
            go.Bar(
                x=daily_counts['Date'],
                y=daily_counts['LLM'],
                name='LLM',
                marker_color='rgba(31, 119, 180, 0.8)'
            )
        )
        
        time_fig.add_trace(
            go.Bar(
                x=daily_counts['Date'],
                y=daily_counts['RAG+LLM'],
                name='RAG+LLM',
                marker_color='rgba(255, 127, 14, 0.8)'
            )
        )
        
        time_fig.add_trace(
            go.Bar(
                x=daily_counts['Date'],
                y=daily_counts['NESSUNA DIFFERENZA'],
                name='Nessuna Differenza',
                marker_color='rgba(44, 160, 44, 0.8)'
            )
        )
        
        time_fig.update_layout(
            title='Confronti per giorno',
            xaxis_title='Data',
            yaxis_title='Numero di confronti',
            barmode='stack',
            bargap=0.15,
            bargroupgap=0.1,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # 3. Percentuali per modello AI
        model_stats = df.groupby(['Modello', 'Vincitore']).size().unstack(fill_value=0).reset_index()
        if 'LLM' not in model_stats.columns:
            model_stats['LLM'] = 0
        if 'RAG+LLM' not in model_stats.columns:
            model_stats['RAG+LLM'] = 0
        if 'NESSUNA DIFFERENZA' not in model_stats.columns:
            model_stats['NESSUNA DIFFERENZA'] = 0
            
        model_stats['Total'] = model_stats['LLM'] + model_stats['RAG+LLM'] + model_stats['NESSUNA DIFFERENZA']
        model_stats['LLM_PCT'] = (model_stats['LLM'] / model_stats['Total']) * 100
        model_stats['RAG_PCT'] = (model_stats['RAG+LLM'] / model_stats['Total']) * 100
        model_stats['TIE_PCT'] = (model_stats['NESSUNA DIFFERENZA'] / model_stats['Total']) * 100
        
        model_fig = go.Figure()
        
        model_fig.add_trace(
            go.Bar(
                x=model_stats['Modello'],
                y=model_stats['LLM_PCT'],
                name='LLM %',
                marker_color='rgba(31, 119, 180, 0.8)',
                text=model_stats['LLM_PCT'].round(1).astype(str) + '%',
                textposition='auto'
            )
        )
        
        model_fig.add_trace(
            go.Bar(
                x=model_stats['Modello'],
                y=model_stats['RAG_PCT'],
                name='RAG+LLM %',
                marker_color='rgba(255, 127, 14, 0.8)',
                text=model_stats['RAG_PCT'].round(1).astype(str) + '%',
                textposition='auto'
            )
        )
        
        model_fig.add_trace(
            go.Bar(
                x=model_stats['Modello'],
                y=model_stats['TIE_PCT'],
                name='Nessuna Differenza %',
                marker_color='rgba(44, 160, 44, 0.8)',
                text=model_stats['TIE_PCT'].round(1).astype(str) + '%',
                textposition='auto'
            )
        )
        
        model_fig.update_layout(
            title='Percentuali di successo per modello AI',
            xaxis_title='Modello AI',
            yaxis_title='Percentuale (%)',
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return {
            "pie_fig": pie_fig,
            "time_fig": time_fig,
            "model_fig": model_fig
        }
    except Exception as e:
        logger.error(f"Errore durante la creazione dei grafici: {str(e)}")
        
        # Ritorna un grafico con l'errore
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text=f"Errore nella generazione dei grafici: {str(e)}",
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return {"error_fig": fig}

def display_comparison_statistics():
    """
    Crea una sezione dell'interfaccia utente per visualizzare le statistiche delle comparazioni.
    """
    st.markdown("<h3>Statistiche dei Confronti LLM vs RAG+LLM</h3>", unsafe_allow_html=True)
    
    # Carica i dati
    df = load_comparison_data()
    
    if df is None or len(df) == 0:
        st.info("Non ci sono ancora dati sufficienti per generare statistiche. Effettua alcuni confronti tra risposte LLM e RAG+LLM per iniziare a raccogliere dati.")
        return
    
    # Analisi dei dati
    stats = analyze_comparison_data(df)
    
    # Mostra statistiche generali
    st.markdown("### Statistiche Generali")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Totale Confronti", stats["total_comparisons"])
    
    with col2:
        llm_pct = f"{stats['llm_percentage']:.1f}%"
        st.metric("Vittorie LLM", stats["llm_wins"], llm_pct)
    
    with col3:
        rag_pct = f"{stats['rag_percentage']:.1f}%"
        st.metric("Vittorie RAG+LLM", stats["rag_wins"], rag_pct)
    
    with col4:
        tie_pct = f"{stats['tie_percentage']:.1f}%"
        st.metric("Nessuna Differenza", stats["ties"], tie_pct)
    
    # Crea grafici
    charts = create_comparison_statistics_charts(df)
    
    if "pie_fig" in charts:
        st.plotly_chart(charts["pie_fig"], use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "time_fig" in charts:
            st.plotly_chart(charts["time_fig"], use_container_width=True)
    
    with col2:
        if "model_fig" in charts:
            st.plotly_chart(charts["model_fig"], use_container_width=True)
    
    # Mostra ultimi confronti
    with st.expander("Ultimi confronti", expanded=False):
        st.markdown("### Ultimi 10 confronti")
        recent_df = df.sort_values('Timestamp', ascending=False).head(10)
        st.dataframe(
            recent_df[['Timestamp', 'Modello', 'Località', 'Domanda', 'Vincitore']],
            use_container_width=True,
            hide_index=True
        )
    
    # Aggiunge un pulsante per esportare i dati
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📊 Scarica dati in CSV",
        data=csv_data,
        file_name="confronti_llm_rag.csv",
        mime="text/csv",
    )


def compare_responses(llm_response, rag_response, model_name):
    """Confronta due risposte per determinare quale è più informativa e affidabile"""
    try:
        # ESECUZIONE ANALISI AVANZATA
        try:
            # Analisi delle singole risposte
            logger.info("Avvio analisi approfondita delle risposte...")
            
            llm_analysis = analyze_single_response(llm_response, "LLM", model_name)
            rag_analysis = analyze_single_response(rag_response, "RAG+LLM", model_name)
            
            # Calcolo score pesati
            llm_score = calculate_weighted_score(llm_analysis)
            rag_score = calculate_weighted_score(rag_analysis)
            
            # Verifica incrociata
            verification = perform_cross_verification(llm_analysis, rag_analysis, model_name)
            
            # Decisione finale
            if llm_score == rag_score:  # Punteggi esattamente uguali
                decision = "NESSUNA DIFFERENZA"
            elif abs(llm_score - rag_score) < 0.005:  # Margine molto stretto
                # Criteri di desempate
                if llm_analysis.dimension_scores[AnalysisDimension.HALLUCINATION_ROBUSTNESS] > rag_analysis.dimension_scores[AnalysisDimension.HALLUCINATION_ROBUSTNESS]:
                    decision = "LLM"
                elif rag_analysis.dimension_scores[AnalysisDimension.ACCURACY] > llm_analysis.dimension_scores[AnalysisDimension.ACCURACY]:
                    decision = "RAG+LLM"
                else:
                    decision = "RAG+LLM"  # Default per parità (RAG generalmente più affidabile)
            else:
                decision = "LLM" if llm_score > rag_score else "RAG+LLM"
            
            # Formattazione della risposta finale
            if decision == "NESSUNA DIFFERENZA":
                final_answer = "ENTRAMBE LE RISPOSTE HANNO LO STESSO LIVELLO DI DETTAGLIO E AFFIDABILITÀ"
            else:
                final_answer = f"TRA LE RISPOSTE CHE IL SISTEMA HA ANALIZZATO, SI PUO' DIRE CHE LA RISPOSTA PIU' COMPLETA E AFFIDABILE E' STATA COSTRUITA TRAMITE IL SISTEMA {decision}"
            
            # NUOVO: Salvataggio del risultato della comparazione
            current_question = st.session_state.get('custom_question', '')
            if st.session_state.get('current_question') is not None and st.session_state.get('current_question') != 'custom':
                # Se è una domanda predefinita, recuperala dalla lista
                questions = [
                   "Riesci a darmi delle informazioni aggiutive per ogni inquinante che è stato analizzato?",
                   "Quale inquinante tra PM2.5, NO2 e O3 presenta il maggiore rischio per il sistema cardiovascolare ?",
                   "Come si differenziano gli impatti sulla salute dell'esposizione a breve termine rispetto a quella cronica per PM10 e PM2.5?",
                   "Quali sono le tecnologie più promettenti per il monitoraggio in tempo reale degli inquinanti atmosferici nelle città italiane?",
                   "Quali sono le correlazioni tra i cambiamenti climatici e l'andamento delle concentrazioni di O3 e PM10?",
                   "Come si confrontano i limiti normativi italiani per SO2 e NO2 con quelli di altri paesi industrializzati?",
                   "Quali sono le evidenze scientifiche sulla relazione tra esposizione a PM2.5 e l'insorgenza di patologie neurodegenerative?",
                   "Come influiscono i diversi modelli di pianificazione urbana sulle concentrazioni di NO2 e PM10 nelle città?"
                ]
                current_question = questions[st.session_state.get('current_question')]
                
            location = st.session_state.get('location', '')
            region = st.session_state.get('region', '')
            
            # Salva il risultato
            save_comparison_result(current_question, final_answer, model_name, location, region)
            
            # Restituisci sia la risposta testuale che i dati dell'analisi per la visualizzazione
            return {
                "decision": final_answer,
                "llm_analysis": llm_analysis,
                "rag_analysis": rag_analysis,
                "llm_score": llm_score,
                "rag_score": rag_score,
                "verification": verification
            }
                
        except Exception as e:
            logger.error(f"Errore durante l'analisi avanzata: {str(e)}")
            
            # Crea una decisione di fallback in caso di errore
            fallback_decision = "LLM"  # Default fallback
            # Salva il risultato usando la decisione di fallback
            current_question = st.session_state.get('custom_question', '')
            if st.session_state.get('current_question') is not None and st.session_state.get('current_question') != 'custom':
                # Se è una domanda predefinita, recuperala dalla lista
                questions = [
                   "Riesci a darmi delle informazioni aggiutive per ogni inquinante che è stato analizzato?",
                   "Quale inquinante tra PM2.5, NO2 e O3 presenta il maggiore rischio per il sistema cardiovascolare ?",
                   "Come si differenziano gli impatti sulla salute dell'esposizione a breve termine rispetto a quella cronica per PM10 e PM2.5?",
                   "Quali sono le tecnologie più promettenti per il monitoraggio in tempo reale degli inquinanti atmosferici nelle città italiane?",
                   "Quali sono le correlazioni tra i cambiamenti climatici e l'andamento delle concentrazioni di O3 e PM10?",
                   "Come si confrontano i limiti normativi italiani per SO2 e NO2 con quelli di altri paesi industrializzati?",
                   "Quali sono le evidenze scientifiche sulla relazione tra esposizione a PM2.5 e l'insorgenza di patologie neurodegenerative?",
                   "Come influiscono i diversi modelli di pianificazione urbana sulle concentrazioni di NO2 e PM10 nelle città?"
                ]
                current_question = questions[st.session_state.get('current_question')]
                
            location = st.session_state.get('location', '')
            region = st.session_state.get('region', '')
            
            save_comparison_result(current_question, fallback_decision, model_name, location, region)
            
            # In caso di fallback, restituisci solo una decisione di fallback
            fallback_message = "Non è stato possibile completare l'analisi dettagliata. LLM selezionato come predefinito."
            return {
                "decision": fallback_message
            }
                
    except Exception as e:
        logger.error(f"Errore durante il confronto delle risposte: {str(e)}")
        return {
            "decision": f"Errore durante il confronto: {str(e)}. Riprova."
        }
              
# COMPONENTI UI 
def create_section(title, content_function=None):
    """Crea una sezione standard dell'app"""
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown(f"<h2 class='section-title'>{title}</h2>", unsafe_allow_html=True)
    
    if content_function:
        content_function()
    
    st.markdown('</div>', unsafe_allow_html=True)

 # CREAZIONE DEL HEADER E LOGO
def create_header():
    """
    Crea un header semplice ed elegante con logo di alta qualità affiancato al titolo.
    """
    import streamlit as st
    import base64
    
    # CODICE SVG
    svg_code = '''
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 200">
      <!-- Forma astratta che assomiglia a una nuvola (verde più chiaro) -->
      <path d="M120,130
               C90,130 75,115 80,90
               C85,65 120,60 140,75
               C160,50 210,50 230,75
               C270,60 310,80 295,120
               C310,150 270,170 240,150
               C200,180 150,175 120,130"
            fill="#7FD483" fill-opacity="0.9" stroke="none"/>
      
      <!-- Elementi geometrici astratti sovrapposti (verdi più chiari) -->
      <ellipse cx="180" cy="100" rx="110" ry="55" fill="#90E095" fill-opacity="0.5"/>
      <ellipse cx="210" cy="115" rx="90" ry="40" fill="#6BD87A" fill-opacity="0.6"/>
      <path d="M130,90 C160,60 240,70 260,100 C240,120 190,125 130,90" fill="#A2EBB0" fill-opacity="0.7"/>
      
      <!-- Testo AIA in azzurro più chiaro (blu più chiaro) -->
      <g transform="translate(320, 120) rotate(180)" text-anchor="middle">
        <!-- Prima A (V capovolta senza trattino) -->
        <path d="M-80,-40 L-60,10 L-40,-40" fill="none" stroke="#64B5F6" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>
        
        <!-- I -->
        <line x1="0" y1="-40" x2="0" y2="10" stroke="#64B5F6" stroke-width="8" stroke-linecap="round"/>
        
        <!-- Seconda A (V capovolta senza trattino) -->
        <path d="M40,-40 L60,10 L80,-40" fill="none" stroke="#64B5F6" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>
      </g>
    </svg>
    '''
    
    # CODICE SVG IN BASE64
    b64 = base64.b64encode(svg_code.encode('utf-8')).decode('utf-8')
    
    # CREAZIONE DELLA SEZIONE HEADER
    with st.container():
        st.markdown('<div class="header-wrapper">', unsafe_allow_html=True)
        
        # CREAZIONE DELLA SEZIONE LOGO E TITOLO
        st.markdown(f'''
            <div style="display: flex; align-items: center; width: 100%;">
                <div style="flex: 0 0 280px;">
                    <img src="data:image/svg+xml;base64,{b64}" class="logo-img">
                </div>
                <div class="header-col-right">
                    <h1 class="header-title">ANALISI INTELLIGENTE ARIA</h1>
                    <p class="header-subtitle">Analisi smart della qualità dell’aria in tempo reale con il supporto di LLM e RAG</p>
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
       
        st.markdown('</div>', unsafe_allow_html=True)
        
                                                           
# CREAZIONE FOOTER
def create_footer():
        """Crea il footer dell'app"""
        current_year = datetime.now().year
        st.markdown(f"""
        <div class="app-footer">
            <p>© {current_year} Monitoraggio Qualità dell'Aria</p>
            <p>Il programma è stato sviluppato ed ideato da Lorand Gjoshi 
            per la sua Tesi Sperimentale in ingegneria informatica presso
            l'Università Telematica Uninettuno</p>
            <p>Ultimo aggiornamento: {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
        </div>
        """, unsafe_allow_html=True)

# FUNZIONE CHE MOSTRA INFORMAZIONI SUI MODELLI AI 
def display_model_explanation():
    """Mostra spiegazione dei modelli AI"""
    with st.expander("💡 Informazioni sui modelli AI", expanded=False):
        st.markdown("""
        ### Modelli AI disponibili
        
        Questa applicazione utilizza modelli cloud per l'analisi della qualità dell'aria:
        
        #### Modelli Cloud
        I modelli cloud sono accessibili tramite API e offrono prestazioni superiori:
        - **GPT-4O**: Modello avanzato di OpenAI, offre le analisi più dettagliate e precise.
        - **Cohere**: Modello alternativo con buone capacità di comprensione e analisi.
        
        La scelta del modello influisce sulla qualità e profondità dell'analisi dei dati sulla qualità dell'aria.
        """)

#FUNZIONE CHE CREA UNA CARD METRICA
def create_metric_card(title, value, unit, description=None, icon=None, color=None, details=None):
    """Crea una card metrica con o senza expander per dettagli"""
    color_style = f"color: {color};" if color else ""
    icon_html = f'<span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>' if icon else ''
    
    card_html = f"""
    <div class="metric-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h4 style="margin: 0;">{icon_html}{title}</h4>
        </div>
        <p class="metric-value" style="{color_style}">{value}<span class="metric-unit">{unit}</span></p>
        {f'<p class="metric-label">{description}</p>' if description else ''}
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    if details:
        with st.expander("Mostra dettagli", expanded=False):
            st.markdown(details)
            
            
# FUNZIONE CHE MOSTRA LE CARD DEGLI INQUINANTI
def pollutants_content():
    """Mostra card per gli inquinanti con template riutilizzabile"""
    components = st.session_state.air_data['list'][0]['components']
    # Definizione info per ogni inquinante
    pollutant_info = {
        "pm2_5": {"name": "PM2.5", "unit": "μg/m³", "icon": "🔬", "color": "#ff6b6b", "description": "Particolato fine"},
        "pm10": {"name": "PM10", "unit": "μg/m³", "icon": "💨", "color": "#f06595", "description": "Particolato grossolano"},
        "no2": {"name": "NO₂", "unit": "μg/m³", "icon": "🚗", "color": "#cc5de8", "description": "Biossido di azoto"},
        "o3": {"name": "O₃", "unit": "μg/m³", "icon": "☀️", "color": "#ffb400", "description": "Ozono"},
        "so2": {"name": "SO₂", "unit": "μg/m³", "icon": "🏭", "color": "#5c7cfa", "description": "Biossido di zolfo"},
        "co": {"name": "CO", "unit": "mg/m³", "icon": "🔥", "color": "#20c997", "description": "Monossido di carbonio"}
    }
    
    cols = st.columns(3)
    
    # TEMPLATE CARDS
    card_template = """
    <div class="metric-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h4 style="margin: 0;"><span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>{name}</h4>
        </div>
        <p class="metric-value" style="color: {color};">{value:.2f}<span class="metric-unit">{unit}</span></p>
        <p class="metric-label">{description}</p>
    </div>
    """
    
    # DETTAGLI INQUINANTI
    details = {
        "pm2_5": "Particolato fine con diametro inferiore a 2.5 micron. Limite UE: 25 μg/m³, OMS: 5 μg/m³",
        "pm10": "Particolato con diametro inferiore a 10 micron. Limite UE: 40 μg/m³, OMS: 15 μg/m³",
        "no2": "Biossido di azoto, gas irritante. Limite UE: 40 μg/m³, OMS: 10 μg/m³",
        "o3": "Ozono troposferico, forte ossidante. Limite UE: 120 μg/m³, OMS: 100 μg/m³",
        "so2": "Biossido di zolfo, gas irritante. Limite UE: 125 μg/m³, OMS: 40 μg/m³",
        "co": "Monossido di carbonio, riduce capacità del sangue. Limite UE: 10 mg/m³, OMS: 4 mg/m³"
    }
    
    # RENDER CARD 
    for i, key in enumerate(["pm2_5", "pm10", "no2", "o3", "so2", "co"]):
        col_idx = i % 3  # Distribuisci su 3 colonne
        with cols[col_idx]:
            info = pollutant_info[key]
            value = components[key]
            
            # NORMALIZZAZIONE CO
            if key == "co" and value > 100:
                value = value / 1000
                
            # RENDER CARD TRAMITE TEMPLATE
            card_html = card_template.format(
                icon=info["icon"],
                name=info["name"],
                color=info["color"],
                value=value,
                unit=info["unit"],
                description=info["description"]
            )
            st.markdown(card_html, unsafe_allow_html=True)
            
            # AGGIUNTA EXPANDER PER DETTAGLI
            with st.expander("Mostra dettagli", expanded=False):
                st.markdown(details[key])
                
# FUNZIONE CHE CREA UN GAUGE PER L'AQI
def create_aqi_gauge(aqi_value):
    """Crea un gauge per visualizzare l'AQI"""
    aqi_description = AQI_DESCRIPTIONS[aqi_value-1]
    aqi_color = AQI_COLORS[aqi_value-1]
    indicator_color = "#00FFFF"  # Azzurro/ciano
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_value,
        title={"text": f"Qualità dell'Aria: <b>{aqi_description}</b>", "font": {"size": 18}},
        gauge={
            'axis': {
                'range': [None, 6], 
                'tickvals': [1, 2, 3, 4, 5, 6],
                'ticktext': ["Buona", "Discreta", "Media", "Scadente", "Molto Scadente", "Pessima"],
                'tickfont': {'size': 10}
            },
            'bar': {'color': "rgba(0,0,0,0)"},  # Rendiamo la barra di progresso trasparente
            'steps': [
                {'range': [0, 1.5], 'color': AQI_COLORS[0]},
                {'range': [1.5, 2.5], 'color': AQI_COLORS[1]},
                {'range': [2.5, 3.5], 'color': AQI_COLORS[2]},
                {'range': [3.5, 4.5], 'color': AQI_COLORS[3]},
                {'range': [4.5, 5.5], 'color': AQI_COLORS[4]},
                {'range': [5.5, 6], 'color': AQI_COLORS[5]}
            ],
            'threshold': {
                'line': {'color': indicator_color, 'width': 10},
                'thickness': 0.85,
                'value': aqi_value
            }
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#444", "family": "Open Sans"}
    )
    return fig

# PAGINA PRINCIPALE DELL'APP
def create_home_page():
    """Crea la pagina principale dell'app"""
    def search_form_content():
     
     submitted = False
    
     
     with st.form("search_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            location = st.text_input(
                "Località", 
                placeholder="Inserisci la tua città...", 
                help="Inserisci il nome della città o località"
            )
        
        with col2:
            region = st.selectbox(
                "Regione", 
                options=[""] + sorted(list(VALID_REGIONS)),
                help="Seleziona la regione in cui si trova la località"
            )
        
        # SEZIONE MODELLO AI
        st.markdown("<h3>Seleziona il modello AI per l'analisi</h3>", unsafe_allow_html=True)
        
        # OPZIONI MENU' A TENDINA
        model_options = {
             "github": " GPT-4O - Perfetto per analisi avanzate",
             "cohere": " Cohere -  Perfetto per una comprensione solida",
             
        }
        
        try:
            current_option_index = list(model_options.keys()).index(st.session_state.get('selected_model_key', 'tinyllama'))
        except (ValueError, KeyError):
            current_option_index = 0
        
        selected_option = st.selectbox(
            "Modello AI",
            options=list(model_options.values()),
            index=current_option_index,
            help="Seleziona il modello AI da utilizzare per l'analisi"
        )
        
        # OTTIENE CHIAVE DEL MODELLO SELEZIONATO 
        selected_model_key = list(model_options.keys())[list(model_options.values()).index(selected_option)]
        selected_model_display = selected_option.split(" - ")[0]
        
        if st.session_state.get('selected_model_key', '') != selected_model_key:
            st.session_state.selected_model_key = selected_model_key
            st.session_state.selected_model_display = selected_model_display
        
        submitted = st.form_submit_button(
            "🔍 Analizza Qualità dell'Aria", 
            type="primary", 
            use_container_width=True,
            help="Avvia l'analisi della qualità dell'aria"
        )
    
     # INFO MODELLI
     display_model_explanation()
    
     if submitted:
        handle_search_submit(location, region, st.session_state.get('selected_model_key', 'tinyllama'))
    
    def info_content():
        st.markdown("""
        ### Cos'è l'Indice di Qualità dell'Aria (AQI)?
        
        L'Indice di Qualità dell'Aria (AQI) è un indicatore che semplifica la comprensione dei livelli di inquinamento:
        """)
        
        # TABELLA AQI
        col1, col2 = st.columns(2)
        
        with col1:
            for i in range(0, 3):
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="width: 30px; height: 30px; border-radius: 50%; background-color: {AQI_COLORS[i]}; margin-right: 10px;"></div>
                    <div>
                        <strong>{i+1} - {AQI_DESCRIPTIONS[i]}</strong><br>
                        <small style="color: var(--text-color);">
                            {["Qualità dell'aria ottima, rischio di inquinamento minimo.", 
                              "Qualità dell'aria accettabile, ma alcuni inquinanti possono destare preoccupazione per persone sensibili.",
                              "Membri di gruppi sensibili possono risentire degli effetti sulla salute."][i]}
                        </small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            for i in range(3, 6):
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="width: 30px; height: 30px; border-radius: 50%; background-color: {AQI_COLORS[i]}; margin-right: 10px;"></div>
                    <div>
                        <strong>{i+1} - {AQI_DESCRIPTIONS[i]}</strong><br>
                        <small style="color: var(--text-color);">
                            {["Tutti possono iniziare a risentire degli effetti sulla salute.",
                              "Avvisi sanitari, tutta la popolazione può risentire di effetti sulla salute.",
                              "Emergenza sanitaria: tutta la popolazione risente di effetti sulla salute."][i-3]}
                        </small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # FUNZIONE DI MANIPOLAZIONE DELLA RICERCA 
    def handle_search_submit(location, region, selected_model):
        # Validazione input
        if not location:
            st.error("Per favore, inserisci una località")
            return
        
        if not region:
            st.error("Per favore, seleziona una regione")
            return
            
        if not validate_input(location, "Località"):
            return
        
        
        region = normalize_region(region, location)
        
        
        st.session_state.model_name = selected_model
        
        # ESECIZONE DELLA RICERCA
        with st.status("Analisi in corso...", expanded=True) as status:
            # GEOLOCALIZZAZIONE
            st.write("🔍 Ricerca coordinate geografiche...")
            coords = get_coordinates_osm(location, region)
            if not coords or not all(coords):
                status.update(label="❌ Analisi fallita: coordinate non trovate", state="error")
                return
                
            lat, lon = coords
            st.write(f"📍 Coordinate: {lat:.4f}, {lon:.4f}")
            
            # DATI SULLA QUALITA' DELL'ARIA ATTUALE
            st.write("🌬️ Recupero dati sulla qualità dell'aria...")
            air_data = get_air_pollution(lat, lon)
            if not air_data:
                status.update(label="❌ Analisi fallita: dati qualità aria non disponibili", state="error")
                return
                
            st.write("✅ Dati qualità aria ricevuti")
            
            # PREVISIONI METEO
            st.write("🌤️ Recupero previsioni meteo...")
            weather_forecast = get_weather_forecast(lat, lon)
            if not weather_forecast:
                st.warning("⚠️ Previsioni meteo non disponibili")
            else:
                st.write("✅ Previsioni meteo ricevute")
                
            # PREVISIONI QUALITYA' DELL'ARIA 
            st.write("📊 Recupero previsioni sulla qualità dell'aria...")
            air_forecast = get_air_pollution_forecast(lat, lon)
            if not air_forecast:
                st.warning("⚠️ Previsioni qualità aria non disponibili")
            else:
                st.write("✅ Previsioni qualità aria ricevute")
            
            # SALVATAGGIO DEI DATI
            st.session_state.air_data_original = air_data.copy() if air_data else None
            
            # SALVATAGGIO DATI NEL CSV 
            st.write("💾 Salvataggio dati nel database locale...")
            saved_data_df = save_data_to_csv(air_data, location, region, lat, lon)
            
            # AGGIORNAMENTO DEL DATABASE VETTORIALE
            if saved_data_df is not None and 'rag_index' in st.session_state and st.session_state.get('rag_index') is not None:
                st.write("🧠 Aggiunta dei dati attuali al database vettoriale ChromaDB...")
                success = add_air_quality_to_index(
                  st.session_state.get('rag_index'),
                  air_data, 
                  location, 
                  region
                )
                if success:
                    st.write("✅ Database vettoriale ChromaDB aggiornato con i nuovi dati")
                else:
                    st.warning("⚠️ Impossibile aggiornare il database vettoriale ChromaDB")
            
            if saved_data_df is not None:
                # ESTRAE I DATI DAL CSV PER EFFETTUARE L'ANALISI PER NON FARE UNA CHIAMATA API ULTERIORE
                components_from_csv = {
                    "co": saved_data_df.iloc[0]["CO"],
                    "no2": saved_data_df.iloc[0]["NO2"],
                    "so2": saved_data_df.iloc[0]["SO2"],
                    "o3": saved_data_df.iloc[0]["O3"],  # Aggiunto O3
                    "pm2_5": saved_data_df.iloc[0]["PM2.5"],
                    "pm10": saved_data_df.iloc[0]["PM10"]
                }
                
                air_data_from_csv = {
                    'list': [{
                        'main': {'aqi': int(saved_data_df.iloc[0]["AQI"])},
                        'components': components_from_csv
                    }]
                }
                
                # UTILIZZO DEI DATI CSV PER L'ANALISI
                st.write("📊 Analisi basata sui dati locali...")
                air_data = air_data_from_csv
                st.write("✅ Dati preparati per l'analisi")
            else:
                st.warning("⚠️ Impossibile salvare i dati nel CSV, utilizzo dei dati originali per l'analisi")
            
            status.update(label="✅ Analisi completata", state="complete")
            
            # SALVATAGGIO DATI NELL'OGGETTO SESSION_STATE
            st.session_state.air_data = air_data
            st.session_state.weather_forecast = weather_forecast
            st.session_state.air_forecast = air_forecast
            st.session_state.location = location
            st.session_state.region = region
            st.session_state.lat = lat
            st.session_state.lon = lon
            
        
            st.session_state.page = 'results'
            
            
            st.rerun()

    
    create_section("📍 Ricerca località", search_form_content)
    create_section("ℹ️ Informazioni generali sui dati della Qualità dell'Aria", info_content)

# PAGINA DEI RISULTATI 
def create_results_page():
    """Crea la pagina dei risultati dell'analisi della qualità dell'aria"""
    
    if ('air_data' not in st.session_state or 
        'location' not in st.session_state or
        'region' not in st.session_state):
        st.error("Dati non disponibili. Torna alla pagina principale per eseguire una nuova analisi.")
        if st.button("Torna alla Pagina Principale", type="primary"):
            st.session_state.page = 'home'
            st.rerun()
        return
    
    # RECUPERO DATI DALLA SESSION_STATE
    air_data = st.session_state.get('air_data')
    weather_forecast = st.session_state.get('weather_forecast')
    air_forecast = st.session_state.get('air_forecast')
    location = st.session_state.get('location', '')
    region = st.session_state.get('region', '')
    lat = st.session_state.get('lat')
    lon = st.session_state.get('lon')
    model_name = st.session_state.get('model_name', 'tinyllama')
    
    # DEFINIZIONI DELLE DESCRIZIONI AQI
    model_display_name = {
        "tinyllama": "TinyLlama",
        "llama2": "Llama2",
        "phi": "Phi",
        "cohere": "Cohere API (Cloud)",
        "github": "GitHub/GPT-4o (Cloud)"
    }.get(model_name, model_name)
    
    components = air_data['list'][0]['components']
    aqi_value = air_data['list'][0]['main']['aqi']
    
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    col_title, col_btn = st.columns([4, 1])
    
    with col_title:
        st.markdown(f"<h1 class='section-title'>Qualità dell'Aria: {location}, {region}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p>Analisi effettuata con {model_display_name} - {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>", unsafe_allow_html=True)
    
    with col_btn:
        if st.button("🔍 Nuova ricerca", type="primary", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
    
    # CREAZIONE DELLA SEZIONE AQI E MAPPA 
    col_aqi, col_map = st.columns([1, 1])
    
    with col_aqi:
        aqi_fig = create_aqi_gauge(aqi_value)
        st.plotly_chart(aqi_fig, use_container_width=True)
        
        # DESCRIZIONE AQI
        aqi_description = AQI_DESCRIPTIONS[aqi_value-1]
        st.info(f"""
        **Livello {aqi_value} - {aqi_description}**
        
        {["L'aria è considerata soddisfacente e l'inquinamento atmosferico comporta rischi minimi o nulli per la salute.",
          "La qualità dell'aria è accettabile, ma alcuni inquinanti possono destare preoccupazione per un ristretto numero di persone particolarmente sensibili.",
          "I membri di gruppi sensibili possono risentire di effetti sulla salute. Il pubblico in generale non è a rischio.",
          "Ogni persona può iniziare a risentire di effetti sulla salute. I membri di gruppi sensibili possono manifestare effetti più gravi.",
          "Avvisi sanitari sulle condizioni di emergenza. È probabile che l'intera popolazione risenta di effetti sulla salute.",
          "Allarme sanitario: tutti possono avere effetti sulla salute più gravi."][aqi_value-1]}
        """)
    
    with col_map:
        # MAPPA INTERATTIVA
        st.markdown("<h3>Mappa della località</h3>", unsafe_allow_html=True)

        m = folium.Map(location=[lat, lon], zoom_start=14, tiles="CartoDB positron")
        
        # AGGIUNTA MARKER E CERCHIO
        popup_html = f"""
        <div style="width: 200px; text-align: center;">
            <h4 style="margin: 5px 0;">{location}, {region}</h4>
            <p style="margin: 5px 0;">AQI: <b>{aqi_value}</b> ({AQI_DESCRIPTIONS[aqi_value-1]})</p>
            <p style="margin: 5px 0;">PM2.5: <b>{components['pm2_5']:.1f} μg/m³</b></p>
            <p style="margin: 5px 0;">PM10: <b>{components['pm10']:.1f} μg/m³</b></p>
            <p style="margin: 5px 0;">O3: <b>{components['o3']:.1f} μg/m³</b></p>
        </div>
        """
        
        marker_color = ["blue", "green", "orange", "yellow", "red", "purple"][aqi_value-1]
        
        folium.Marker(
            [lat, lon], 
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{location} - AQI: {aqi_value}",
            icon=folium.Icon(color=marker_color, icon="info-sign")
        ).add_to(m)
        
        folium.Circle(
            radius=2000,
            location=[lat, lon],
            color=AQI_COLORS[aqi_value-1],
            fill=True,
            fill_opacity=0.2
        ).add_to(m)
        
        folium_html = m._repr_html_()
        st_html(folium_html, height=300)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # AGGIUNTA: RIQUADRO CON CARD DEGLI INQUINANTI SOTTO GAUGE E MAPPA
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown("<h3>Inquinanti principali</h3>", unsafe_allow_html=True)
    
    # Definizione della funzione display_pollutants_cards se non è nel contesto globale
    def display_pollutants_cards_local(components):
        """Mostra card per gli inquinanti con template riutilizzabile"""
        # Definizione info per ogni inquinante
        pollutant_info = {
            "pm2_5": {"name": "PM2.5", "unit": "μg/m³", "icon": "🔬", "color": "#ff6b6b", "description": "Particolato fine"},
            "pm10": {"name": "PM10", "unit": "μg/m³", "icon": "💨", "color": "#f06595", "description": "Particolato grossolano"},
            "no2": {"name": "NO₂", "unit": "μg/m³", "icon": "🚗", "color": "#cc5de8", "description": "Biossido di azoto"},
            "o3": {"name": "O₃", "unit": "μg/m³", "icon": "☀️", "color": "#ffb400", "description": "Ozono"},
            "so2": {"name": "SO₂", "unit": "μg/m³", "icon": "🏭", "color": "#5c7cfa", "description": "Biossido di zolfo"},
            "co": {"name": "CO", "unit": "mg/m³", "icon": "🔥", "color": "#20c997", "description": "Monossido di carbonio"}
        }
        
        cols = st.columns(3)
        
        # TEMPLATE CARDS
        card_template = """
        <div class="metric-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0;"><span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>{name}</h4>
            </div>
            <p class="metric-value" style="color: {color};">{value:.2f}<span class="metric-unit">{unit}</span></p>
            <p class="metric-label">{description}</p>
        </div>
        """
        
        # DETTAGLI INQUINANTI
        details = {
            "pm2_5": "Particolato fine con diametro inferiore a 2.5 micron. Limite UE: 25 μg/m³, OMS: 5 μg/m³",
            "pm10": "Particolato con diametro inferiore a 10 micron. Limite UE: 40 μg/m³, OMS: 15 μg/m³",
            "no2": "Biossido di azoto, gas irritante. Limite UE: 40 μg/m³, OMS: 10 μg/m³",
            "o3": "Ozono troposferico, forte ossidante. Limite UE: 120 μg/m³, OMS: 100 μg/m³",
            "so2": "Biossido di zolfo, gas irritante. Limite UE: 125 μg/m³, OMS: 40 μg/m³",
            "co": "Monossido di carbonio, riduce capacità del sangue. Limite UE: 10 mg/m³, OMS: 4 mg/m³"
        }
        
        # RENDER CARD 
        for i, key in enumerate(["pm2_5", "pm10", "no2", "o3", "so2", "co"]):
            col_idx = i % 3  # Distribuisci su 3 colonne
            with cols[col_idx]:
                info = pollutant_info[key]
                value = components[key]
                
                # NORMALIZZAZIONE CO
                if key == "co" and value > 100:
                    value = value / 1000
                    
                # RENDER CARD TRAMITE TEMPLATE
                card_html = card_template.format(
                    icon=info["icon"],
                    name=info["name"],
                    color=info["color"],
                    value=value,
                    unit=info["unit"],
                    description=info["description"]
                )
                st.markdown(card_html, unsafe_allow_html=True)
                
                # AGGIUNTA EXPANDER PER DETTAGLI
                with st.expander("Mostra dettagli", expanded=False):
                    st.markdown(details[key])
    
    # Chiamare la funzione locale o quella globale a seconda di quale è disponibile
    display_pollutants_cards_local(components)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # DEFINIZIONE FUNZIONI DI CONTENUTO
    def pollutants_content():
        """Mostra card per gli inquinanti e grafico"""
    
        # GRAFICO A BARRE
        st.markdown("<h3>Confronto inquinanti</h3>", unsafe_allow_html=True)
        values = [
            float(components.get('pm2_5', 0) or 0), 
            float(components.get('pm10', 0) or 0), 
            float(components.get('no2', 0) or 0), 
            float(components.get('o3', 0) or 0),  
            float(components.get('so2', 0) or 0), 
            float(components.get('co', 0)/1000 if components.get('co', 0) > 100 else components.get('co', 0) or 0)
        ]
    
        df_pollutants = pd.DataFrame({
            'Inquinante': ['PM2.5', 'PM10', 'NO₂', 'O₃', 'SO₂', 'CO'],
            'Valore': values,
            'Unità': ['μg/m³', 'μg/m³', 'μg/m³', 'μg/m³', 'μg/m³', 'mg/m³']
        })
    
        # go.Figure PER CONTROLLO PIU' PRECISO
        fig = go.Figure()
    
        # COLORI INQUINANTI
        colors = {
            'PM2.5': '#ff6b6b', 
            'PM10': '#f06595', 
            'NO₂': '#cc5de8', 
            'O₃': '#ffb400', 
            'SO₂': '#5c7cfa', 
            'CO': '#20c997'
        }
    
        for i, row in df_pollutants.iterrows():
            fig.add_trace(go.Bar(
                x=[row['Inquinante']], 
                y=[row['Valore']],
                name=row['Inquinante'],
                marker_color=colors[row['Inquinante']],
                text=[f"{row['Valore']:.2f}"],
                textposition='outside',
                width=0.6  # Larghezza della barra
            ))
    
        # LAYOUT GRAFICO
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=30, b=100),
            xaxis=dict(tickangle=0),
            yaxis=dict(title="Concentrazione"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True,  # Mostra legenda
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            barmode='group'
        )
    
        # LINEE DI RIFERIMENTO OMS
        who_limits = {
            'PM2.5': 5,
            'PM10': 15,
            'NO₂': 10,
            'O₃': 100,
            'SO₂': 40,
            'CO': 4
        }
    
        # LINEE DI RIFERIMENTO OMS
        for i, pollutant in enumerate(df_pollutants['Inquinante']):
            if pollutant in who_limits:
                fig.add_shape(
                    type="line",
                    x0=i-0.4, x1=i+0.4,
                    y0=who_limits[pollutant], y1=who_limits[pollutant],
                    line=dict(color="red", width=2, dash="dash")
                )
    
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
        # LEGENDA GRAFICO
        st.markdown("""
            <div style="text-align: center; margin-top: -20px;">
                <span style="color: red; margin-right: 5px;">- - -</span> <small>Linea tratteggiata = limite raccomandato OMS</small>
            </div>
            """, unsafe_allow_html=True)
    
    def forecasts_content():
        """Mostra previsioni meteo e qualità aria fianco a fianco"""
    
        st.markdown("<h3>Previsioni per oggi e domani</h3>", unsafe_allow_html=True)
    
        col_weather, col_air = st.columns(2)
    
        with col_weather:
            st.markdown("<h4 style='text-align: center;'>Previsioni Meteo</h4>", unsafe_allow_html=True)
            
            if weather_forecast and 'forecasts' in weather_forecast and weather_forecast['forecasts']:
                
                forecast_by_day = {}
                for fc in weather_forecast['forecasts']:
                    day = fc['giorno']
                    if day not in forecast_by_day:
                        forecast_by_day[day] = []
                    forecast_by_day[day].append(fc)
                
                day_labels = ["Oggi", "Domani"]
                
                for day in range(2):  
                    if day in forecast_by_day and forecast_by_day[day]:
                        forecasts = forecast_by_day[day]
                        
                        avg_temp = sum(fc['temperatura'] for fc in forecasts) / len(forecasts)
                        min_temp = min(fc['temperatura'] for fc in forecasts)
                        max_temp = max(fc['temperatura'] for fc in forecasts)
                        avg_humidity = sum(fc['umidita'] for fc in forecasts) / len(forecasts)
                        avg_wind = sum(fc['vento_velocita'] for fc in forecasts) / len(forecasts)
                        max_rain_prob = max(fc['probabilita_pioggia'] for fc in forecasts)
                        
                        conditions = [fc['condizioni'] for fc in forecasts]
                        main_condition = max(set(conditions), key=conditions.count)
                        
                        icon_code = next((fc['icona'] for fc in forecasts if fc['condizioni'] == main_condition), "01d")
                        icon_url = f"https://openweathermap.org/img/wn/{icon_code}@2x.png"
                        
                        # CREAZIONE CARD CON CSS
                        st.markdown(f"""
                        <div class="forecast-card">
                            <h4 class="forecast-title">{day_labels[day]}</h4>
                            <div class="forecast-icon-container">
                                <img src="{icon_url}" alt="{main_condition}" style="width: 40px; height: 40px;" />
                                <div>
                                    <p style="font-size: 1.2rem; margin: 0;">{avg_temp:.1f}°C</p>
                                    <p style="margin: 0; font-size: 0.8rem;">{main_condition.title()}</p>
                                </div>
                            </div>
                            <div class="forecast-data-container">
                                <div class="forecast-data-column">
                                    <p class="forecast-data-row"><strong>Min/Max:</strong> {min_temp:.1f}°C / {max_temp:.1f}°C</p>
                                    <p class="forecast-data-row"><strong>Umidità:</strong> {avg_humidity:.0f}%</p>
                                </div>
                                <div class="forecast-data-column">
                                    <p class="forecast-data-row"><strong>Vento:</strong> {avg_wind:.1f} m/s</p>
                                    <p class="forecast-data-row"><strong>Prob. pioggia:</strong> {max_rain_prob:.0f}%</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Previsioni meteo non disponibili")
    
        with col_air:
            st.markdown("<h4 style='text-align: center;'>Previsioni Qualità Aria</h4>", unsafe_allow_html=True)
            
            if air_forecast and 'forecasts' in air_forecast and air_forecast['forecasts']:
                
                day_forecasts = {}
                for fc in air_forecast['forecasts']:
                    day = fc['giorno']
                    if day not in day_forecasts:
                        day_forecasts[day] = []
                    day_forecasts[day].append(fc)
                
                day_labels = ["Oggi", "Domani"]
                
                for day in range(2): 
                    if day in day_forecasts and day_forecasts[day]:
                        forecasts = day_forecasts[day]
                        
                        avg_aqi = sum(fc['aqi'] for fc in forecasts) / len(forecasts)
                        avg_aqi_int = int(round(avg_aqi))
                        avg_pm25 = sum(fc['pm2_5'] for fc in forecasts) / len(forecasts)
                        avg_pm10 = sum(fc['pm10'] for fc in forecasts) / len(forecasts)
                        avg_o3 = sum(fc['o3'] for fc in forecasts) / len(forecasts)
                        avg_no2 = sum(fc['no2'] for fc in forecasts) / len(forecasts)
                        avg_so2 = sum(fc['so2'] for fc in forecasts) / len(forecasts)
                        avg_co = sum(fc['co'] for fc in forecasts) / len(forecasts)
                        
                        if avg_co > 100:
                            avg_co = avg_co / 1000
                            co_unit = "mg/m³"
                        else:
                            co_unit = "μg/m³"
                        
                        # DESCRIZIONE AQI
                        aqi_text = AQI_DESCRIPTIONS[avg_aqi_int-1] if 1 <= avg_aqi_int <= len(AQI_DESCRIPTIONS) else "Non disponibile"
                        aqi_color = AQI_COLORS[avg_aqi_int-1] if 1 <= avg_aqi_int <= len(AQI_COLORS) else "#cccccc"
                        
                        aqi_icons = ["😊", "🙂", "😐", "😷", "⚠️", "☣️"]
                        aqi_icon = aqi_icons[avg_aqi_int-1] if 1 <= avg_aqi_int <= len(aqi_icons) else "❓"
                        
                        # CREAZIONE CARD CON CSS
                        st.markdown(f"""
                        <div class="forecast-card">
                            <h4 class="forecast-title">{day_labels[day]}</h4>
                            <div class="forecast-icon-container">
                                <div style="font-size: 1.5rem; margin-right: 10px;">{aqi_icon}</div>
                                <div>
                                    <p style="background-color: {aqi_color}; color: white; padding: 3px; border-radius: 5px; margin: 0; font-size: 0.9rem;">
                                        AQI: <strong>{avg_aqi:.1f}</strong>
                                    </p>
                                    <p style="margin: 0; font-size: 0.8rem;">{aqi_text}</p>
                                </div>
                            </div>
                            <div class="forecast-data-container">
                                <div class="forecast-data-column">
                                    <p class="forecast-data-row"><strong>PM2.5:</strong> {avg_pm25:.1f} μg/m³</p>
                                    <p class="forecast-data-row"><strong>PM10:</strong> {avg_pm10:.1f} μg/m³</p>
                                    <p class="forecast-data-row"><strong>O3:</strong> {avg_o3:.1f} μg/m³</p>
                                </div>
                                <div class="forecast-data-column">
                                    <p class="forecast-data-row"><strong>NO₂:</strong> {avg_no2:.1f} μg/m³</p>
                                    <p class="forecast-data-row"><strong>SO₂:</strong> {avg_so2:.1f} μg/m³</p>
                                    <p class="forecast-data-row"><strong>CO:</strong> {avg_co:.2f} {co_unit}</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Previsioni qualità aria non disponibili")
    
    def ai_analysis_content():
        """Crea la sezione di analisi intelligente con doppia risposta"""
        
        if 'question_answers' not in st.session_state:
            st.session_state.question_answers = {}
        if 'custom_question' not in st.session_state:
            st.session_state.custom_question = ""
        if 'current_question' not in st.session_state:
            st.session_state.current_question = None
        if 'comparison_result' not in st.session_state:
            st.session_state.comparison_result = None
        if 'show_statistics' not in st.session_state:
            st.session_state.show_statistics = False
        
        st.markdown("<h3>Fai una domanda sulla qualità dell'aria</h3>", unsafe_allow_html=True)
        
        with st.form(key="custom_question_form"):
            custom_question = st.text_area(
                "Scrivi qui la tua domanda", 
                height=100, 
                value=st.session_state.get('custom_question', ''),
                placeholder="Es: Quali sono gli effetti del PM2.5 sulla salute? Come influisce il meteo sulla qualità dell'aria?"
            )
            
            custom_submit = st.form_submit_button(
                "🔍 Analizza", 
                use_container_width=True, 
                type="primary"
            )
        
        if custom_submit and custom_question.strip():
            st.session_state.custom_question = custom_question
            st.session_state.current_question = "custom"
            st.session_state.comparison_result = None  # Reset del risultato di comparazione
            st.session_state.show_statistics = False  # Nascondi statistiche quando si fa una nuova domanda
            
            with st.spinner("Elaborazione risposta in corso..."):
                custom_answer = interpret_air_quality(
                    components, 
                    model_name, 
                    custom_question, 
                    weather_forecast, 
                    air_forecast, 
                    st.session_state.get('rag_index')
                )
                st.session_state.question_answers['custom'] = custom_answer
        
        # DOMANDE RAPIDE 
        st.markdown("<h3>Domande rapide</h3>", unsafe_allow_html=True)
        st.markdown("Seleziona una delle seguenti domande per ottenere una risposta immediata:")
        
        # LISTA DOMANDE DISPONIBILI
        questions = [
            "Riesci a darmi delle informazioni aggiutive per ogni inquinante che è stato analizzato?",
            "Quale inquinante tra PM2.5, NO2 e O3 presenta il maggiore rischio per il sistema cardiovascolare ?",
            "Come si differenziano gli impatti sulla salute dell'esposizione a breve termine rispetto a quella cronica per PM10 e PM2.5?",
            "Quali sono le tecnologie più promettenti per il monitoraggio in tempo reale degli inquinanti atmosferici nelle città italiane?",
            "Quali sono le correlazioni tra i cambiamenti climatici e l'andamento delle concentrazioni di O3 e PM10?",
            "Come si confrontano i limiti normativi italiani per SO2 e NO2 con quelli di altri paesi industrializzati?",
            "Quali sono le evidenze scientifiche sulla relazione tra esposizione a PM2.5 e l'insorgenza di patologie neurodegenerative?",
            "Come influiscono i diversi modelli di pianificazione urbana sulle concentrazioni di NO2 e PM10 nelle città?"
        ]
        
        col1, col2 = st.columns(2)
        
        for i, question in enumerate(questions):
            col = col1 if i % 2 == 0 else col2
            with col:
                if st.button(question, key=f"btn_{i}", use_container_width=True, 
                        help="Clicca per analizzare questa domanda"):
                    st.session_state.current_question = i
                    st.session_state.comparison_result = None  
                    st.session_state.show_statistics = False  # Nascondi statistiche quando si fa una nuova domanda
                    
                    with st.spinner("Elaborazione risposta in corso..."):
                        analysis = interpret_air_quality(
                            components,
                            model_name,
                            questions[i],
                            weather_forecast,
                            air_forecast,
                            st.session_state.get('rag_index')
                        )
                        st.session_state.question_answers[i] = analysis
        
        # Bottone reset
        if st.button("RESET E FAI UNA NUOVA DOMANDA", use_container_width=True):
            st.session_state.question_answers = {}
            st.session_state.custom_question = ""
            st.session_state.current_question = None
            st.session_state.comparison_result = None  
            st.session_state.show_statistics = False  # Nascondi statistiche quando si fa reset
            st.info("Tutte le risposte sono state resettate")
        
        if st.session_state.get('current_question') is not None and st.session_state.get('question_answers'):
            st.markdown("<h3>Risposta</h3>", unsafe_allow_html=True)
            
            with st.container():
                current_q = st.session_state.get('current_question')
                
                if current_q == "custom":
                    st.markdown(f"""
                    <div class="question-container">
                        <strong>Domanda:</strong> {st.session_state.get('custom_question', '')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    responses = st.session_state.get('question_answers', {}).get('custom', {"llm": "", "rag": ""})
                else:
                    st.markdown(f"""
                    <div class="question-container">
                        <strong>Domanda:</strong> {questions[current_q]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    responses = st.session_state.get('question_answers', {}).get(current_q, {"llm": "", "rag": ""})
                
                # RISPOSTE LLM
                st.markdown("""
                <div class="answer-section-header">
                    <h4 class="answer-section-title">RISPOSTA TRAMITE LLM</h4>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(responses.get("llm", ""))
                
                st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
                
                # RISPOSTA RAG + LLM
                st.markdown("""
                <div class="answer-section-header">
                    <h4 class="answer-section-title">RISPOSTA TRAMITE RAG+LLM</h4>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(responses.get("rag", ""))
                
                # PULSANTE PER COMPARAZIONE
                st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
                if st.button("MOSTRA QUALE TRA LE DUE RISPOSTE È PIÙ RICCA DI DETTAGLI E SPIEGAZIONI", 
                        use_container_width=True, 
                        key="compare_button"):
                    with st.spinner("Confronto in corso..."):
                        # Ottieni le risposte e il model_name dal contesto attuale
                        current_responses = responses
                        current_model_name = st.session_state.get('model_name', 'github')
                        
                        comparison_result = compare_responses(
                            current_responses.get("llm", ""), 
                            current_responses.get("rag", ""), 
                            current_model_name
                        )
                        st.session_state.comparison_result = comparison_result
                
                # RISULTATO COMPARAZIONE
                if st.session_state.get('comparison_result'):
                    # Mostra la decisione testuale
                    decision_text = st.session_state.comparison_result
                    if isinstance(st.session_state.comparison_result, dict):
                        decision_text = st.session_state.comparison_result.get('decision', '')
                    
                    st.markdown(f"""
                    <div class="comparison-result">
                        <h4 class="comparison-title">{decision_text}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Se abbiamo i dettagli dell'analisi, mostra le visualizzazioni
                    if isinstance(st.session_state.comparison_result, dict) and 'llm_analysis' in st.session_state.comparison_result:
                        try:
                            result = st.session_state.comparison_result
                            
                            # Visualizza il radar chart
                            radar_fig = create_comparison_visualization(
                                result["llm_analysis"], 
                                result["rag_analysis"],
                                result["llm_score"],
                                result["rag_score"]
                            )
                            st.plotly_chart(radar_fig, use_container_width=True)
                            
                            # Visualizza la tabella dei punteggi
                            table_fig = create_comparison_table(
                                result["llm_analysis"], 
                                result["rag_analysis"]
                            )
                            st.plotly_chart(table_fig, use_container_width=True)
                            
                            # Aggiungi una spiegazione del grafico
                            with st.expander("📊 Spiegazione dell'analisi comparativa", expanded=False):
                                st.markdown("""
                                **Interpretazione dell'analisi comparativa:**
                                
                                - **Grafico Radar:** Mostra le otto dimensioni di valutazione. Valori più alti (più esterni) indicano migliori prestazioni in quella dimensione.
                                
                                - **Tabella dei Punteggi:** Confronta i valori normalizzati in percentuale per ogni dimensione e mostra la differenza tra le due risposte.
                                
                                - **Punteggio Finale:** Calcolato come media ponderata delle otto dimensioni, con pesi diversi per riflettere l'importanza relativa di ciascuna dimensione.
                                
                                **Le otto dimensioni analizzate sono:**
                                
                                1. **Accuratezza e Veridicità:** Correttezza fattuale e fedeltà alle fonti
                                   - Factual consistency: correttezza fattuale rispetto alla documentazione
                                   - Justification alignment: allineamento delle argomentazioni con le conclusioni
                                   - Explicit source accuracy: correttezza delle citazioni esplicite
                                
                                2. **Completezza Informativa:** Copertura di tutti gli aspetti rilevanti
                                   - Coverage score: copertura dei concetti chiave rispetto a una risposta ideale
                                   - Content density: densità informativa, rapporto tra contenuti utili e lunghezza
                                
                                3. **Robustezza alle Allucinazioni:** Assenza di invenzioni infondate
                                   - Hallucination rate: frequenza di affermazioni non supportate da evidenze
                                   - Source backed claim ratio: percentuale di affermazioni supportate da fonti
                                
                                4. **Qualità Linguistica:** Espressione chiara e strutturata
                                   - Fluency score: correttezza grammaticale e fluidità del testo
                                   - Coherence score: coerenza logica e strutturale del contenuto
                                
                                5. **Utilità Percepita:** Valore concreto della risposta per l'utente
                                   - User utility: valutazione dell'utilità da parte di valutatori esperti
                                   - Interrogability score: qualità nel stimolare ulteriori domande
                                
                                6. **Accessibilità e Leggibilità:** Facilità di comprensione per qualsiasi utente
                                   - Readability score: indice di leggibilità calcolato
                                   - Technical jargon count: numero di termini tecnici non contestualizzati
                                
                                7. **Collegamento Fonti:** Efficacia nel collegare affermazioni e fonti
                                   - Evidence linking score: efficacia nel collegare affermazioni ed evidenze
                                   - Answer traceability: facilità nel tracciare le fonti delle affermazioni
                                
                                8. **Validità Temporale:** Considerazione appropriata della dimensione temporale
                                   - Temporal validity: correttezza nella gestione degli aspetti temporali
                                """)
                        except Exception as e:
                            st.error(f"Errore nella visualizzazione dei grafici: {str(e)}")
                
                    # Mostra il pulsante per le statistiche solo dopo che è stato premuto il pulsante di comparazione
                    # NUOVO: Pulsante per mostrare/nascondere le statistiche
                    stats_button_text = "NASCONDI STATISTICHE DI CONFRONTO" if st.session_state.get('show_statistics', False) else "MOSTRA STATISTICHE DI CONFRONTO"
                    if st.button(stats_button_text, use_container_width=True, key="stats_button_after_comparison"):
                        st.session_state.show_statistics = not st.session_state.get('show_statistics', False)
                        st.rerun()
                
                # CONTENUTO RAG OPZIONALE 
                if 'rag_context' in st.session_state and st.session_state.get('rag_context'):
                    with st.expander("Mostra contesto utilizzato dal sistema RAG", expanded=False):
                        st.markdown("#### Documenti rilevanti usati per questa analisi:")
                        for i, ctx in enumerate(st.session_state.get('rag_context', [])):
                            st.markdown(f"**Documento {i+1}:**\n{ctx}")
        
        # NUOVO: Visualizzazione statistiche se attivata dopo la comparazione
        if st.session_state.get('show_statistics', False) and st.session_state.get('comparison_result'):
            st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
            display_comparison_statistics()
    
    # CREAZIONE SEZIONI PAGINA
    create_section("📊 Dettagli Inquinanti", pollutants_content)
    create_section("🌱 Previsioni", forecasts_content)
    create_section("🧠 Analisi Intelligente", ai_analysis_content)
    
     
# FUNZIONE PER IL DEBUG DEL SISTEMA RAG 
def debug_rag_index(index):
    """Funzione utile per il debug del sistema RAG"""
    if index is None:
        st.error("Indice non disponibile per il debug")
        return
        
    st.write(f"Tipo di indice: {type(index)}")
    st.write(f"Tipo di vector store: {type(index._vector_store)}")
    
    try:
        
        vector_store = index._vector_store
        chroma_collection = vector_store.chroma_collection
        
    
        try:
            count = chroma_collection.count()
            st.write(f"Numero di documenti nella collezione: {count}")
        except Exception as e:
            st.write(f"Errore nel conteggio documenti: {str(e)}")
        
        
        try:
           
            ids = chroma_collection.get(limit=3)["ids"]
            if ids:
                st.write(f"Esempi di IDs: {ids}")
                
                result = chroma_collection.get(ids=ids[:3], include=["metadatas"])
                if "metadatas" in result and result["metadatas"]:
                    st.write("### Esempi di metadati:")
                    for i, metadata in enumerate(result["metadatas"]):
                        st.write(f"Documento {i+1} (ID: {ids[i]}):")
                        st.json(metadata)
        except Exception as e:
            st.write(f"Errore nell'estrazione dei metadati: {str(e)}")

       
        try:
            if Settings.embed_model:
                st.write(f"Dimensione embed_model: {Settings.embed_model.embed_dimension}")
                st.write(f"Tipo embed_model: {type(Settings.embed_model).__name__}")
        except Exception as e:
            st.write(f"Errore nell'accesso alle informazioni sul modello: {str(e)}")
            
       
        try:
            cache_files = os.listdir(EMBEDDINGS_CACHE_DIR)
            st.write(f"Numero di embedding nella cache: {len(cache_files)}")
            if cache_files:
                with st.expander("Esempi di file nella cache", expanded=False):
                    for i, file in enumerate(cache_files[:5]):  # Mostra solo i primi 5
                        file_path = os.path.join(EMBEDDINGS_CACHE_DIR, file)
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        st.write(f"File {i+1}: {file}")
                        st.write(f"Timestamp: {data.get('timestamp', 'N/A')}")
                        embedding = data.get('embedding', [])
                        st.write(f"Dimensione embedding: {len(embedding)}")
                        st.write(f"Primi 5 valori: {embedding[:5]}...")
        except Exception as e:
            st.write(f"Errore nell'analisi del cache: {str(e)}")
            
    except Exception as e:
        st.write(f"Errore generale durante il debug: {str(e)}")
        
# FUNZIONE PER INIZIALIZZARE LO STATO DELLA SESSIONE
def initialize_session_state():
    """Inizializza tutte le variabili di sessione necessarie"""
    
    # DEFIZNIZIONE DEI VALORI DI DEFAULT 
    default_values = {
    'page': 'home',
    'rag_index': None,
    'model_name': 'github',
    'selected_model_key': 'github',
    'selected_model_display': 'GPT-4O - Cloud',
    'question_answers': {},
    'custom_question': '',
    'current_question': None,
    'rag_context': [],
    'embed_model_configured': False,
    'air_data': None,
    'weather_forecast': None,
    'air_forecast': None,
    'location': None,
    'region': None,
    'lat': None,
    'lon': None,
    'comparison_result': None,  
    'authenticated': False,
    'login_attempts': 0,
    'show_statistics': False
  }
    
    
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


# UNZIONE CHE VERIFICA L'API KEY DI GITHUB/GPT4
def verify_github_api():
    """Verifica se l'API key di GitHub/GPT è configurata e funzionante"""
    try:
        if not GITHUB_TOKEN:
            return False, "Token GitHub non configurato"
        
        # CHIAMATA PROVA
        client = OpenAI(
            base_url=GITHUB_AZURE_ENDPOINT,
            api_key=GITHUB_TOKEN,
        )
        
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Rispondi solo con 'OK'."},
                {"role": "user", "content": "Test connessione API"}
            ],
            model=GITHUB_MODEL_NAME,
            temperature=0.1,
            max_tokens=10,
            top_p=1.0
        )
        
        
        if response and hasattr(response, 'choices') and len(response.choices) > 0:
            return True, "API funzionante"
        else:
            return False, "Risposta API non valida"
    except Exception as e:
        logger.error(f"Errore verifica API GitHub: {str(e)}")
        return False, str(e)

# FUNZIONE CHE VERIFICA L'API KEY DI COHERE 
def verify_cohere_api():
    """Verifica se l'API key di Cohere è configurata e funzionante"""
    try:
        if not COHERE_API_KEY:
            return False, "API key Cohere non configurata"
        
        # CHIAMATA PROVA
        headers = {
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "message": "Test connessione API",
            "model": "command",
            "temperature": 0.1,
            "max_tokens": 10
        }
        
        response = requests.post(COHERE_API_ENDPOINT, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return True, "API funzionante"
        else:
            return False, f"Errore API: status code {response.status_code}"
    except Exception as e:
        logger.error(f"Errore verifica API Cohere: {str(e)}")
        return False, str(e)

# CREAZIONE SIDEBAR
def create_sidebar():
    """Crea la sidebar dell'applicazione con autenticazione e strumenti essenziali per developer"""
    with st.sidebar:
        st.markdown("<h3>Informazioni</h3>", unsafe_allow_html=True)
        
        # VERIFICA SE L'UTENTE E' STATO AUTENTICATO
        if not st.session_state.get('authenticated', False):
            # Limita i tentativi di accesso
            if st.session_state.get('login_attempts', 0) >= 3:
                st.error("Troppi tentativi falliti. Ricarica la pagina per riprovare.")
                return
            
            # FORM DI LOGIN DEVELOPER
            with st.form("sidebar_login_form"):
                st.subheader("Accesso Developer")
                username = st.text_input("Username", key="sidebar_username")
                password = st.text_input("Password", type="password", key="sidebar_password")
                submit = st.form_submit_button("Accedi")
                
                if submit:
                    # CREDENZIALI
                    if username == "admin" and password == "admin123":
                        st.session_state.authenticated = True
                        st.success("Accesso effettuato con successo!")
                        st.rerun()  # Aggiorna la pagina per mostrare la sidebar completa
                    else:
                        st.session_state.login_attempts = st.session_state.get('login_attempts', 0) + 1
                        st.error(f"Credenziali non valide. Tentativo {st.session_state.login_attempts}/3")
            
            # INFO BASE PRE-LOGIN
            st.markdown("""
            Questa applicazione consente di monitorare la qualità dell'aria.
            
            **Accedi come developer per visualizzare:**
            - Statistiche del sistema
            - Gestione della cache
            - Strumenti di diagnostic
            """)
            
            return  # Esci dalla funzione se non autenticato
        
        st.success("👩‍💻 Modalità Developer attiva")
        
        # STATO DEL SISTEMA
        st.markdown("### 🔄 Stato del Sistema")
        
        # VERIFICA CHROMA DB
        try:
            if os.path.exists(VECTOR_DB_DIR):
                st.success("✅ Directory ChromaDB: PRESENTE")
            else:
                st.error(f"❌ Directory ChromaDB: NON TROVATA")
                
            chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
            st.success("✅ Client ChromaDB: CONNESSO")
        except Exception as e:
            st.error(f"❌ Client ChromaDB: ERRORE CONNESSIONE")
        
        # RAG INDEX CHECK
        if st.session_state.get('rag_index'):
            st.success("✅ Indice RAG: ATTIVO")
        else:
            st.error("❌ Indice RAG: NON DISPONIBILE")
            
            if st.button("🚀 Inizializza RAG", key="init_rag"):
                with st.spinner("Inizializzazione RAG in corso..."):
                    try:
                        new_index = load_rag_index()
                        if new_index:
                            st.session_state.rag_index = new_index
                            st.success("✅ Sistema RAG inizializzato con successo")
                            st.rerun()
                        else:
                            st.error("❌ Impossibile inizializzare il sistema RAG")
                    except Exception as e:
                        st.error(f"❌ Errore inizializzazione RAG: {str(e)}")
        
        # STATO EMBEDDING MODEL
        if st.session_state.get('embed_model_configured'):
            st.success("✅ Embedding model: CONFIGURATO")
        else:
            st.error("❌ Embedding model: NON CONFIGURATO")
        
        # VERIFICA DELLE API
        st.markdown("### 🔑 Verifica API")
         
        # Pulsante per GitHub/GPT
        if st.button("🔍 Verifica API GitHub/GPT", key="check_github_api", use_container_width=True):
            with st.spinner("Verifica in corso..."):
                status, message = verify_github_api()
                st.session_state.github_api_status = status
                st.session_state.github_api_message = message

        # Mostra risultato GitHub/GPT se disponibile
        if 'github_api_status' in st.session_state:
            if st.session_state.github_api_status:
                st.success("✅ GitHub/GPT: OK")
            else:
                st.error(f"❌ GitHub/GPT: {st.session_state.get('github_api_message', '')}")

        st.markdown("---")  # Separatore sottile

        # Pulsante per Cohere
        if st.button("🔍 Verifica API Cohere", key="check_cohere_api", use_container_width=True):
            with st.spinner("Verifica in corso..."):
                status, message = verify_cohere_api()
                st.session_state.cohere_api_status = status
                st.session_state.cohere_api_message = message

        # Mostra risultato Cohere se disponibile
        if 'cohere_api_status' in st.session_state:
            if st.session_state.cohere_api_status:
                st.success("✅ Cohere: OK")
            else:
                st.error(f"❌ Cohere: {st.session_state.get('cohere_api_message', '')}")

        # Pulsante reset (opzionale, solo se c'è almeno un risultato)
        if ('github_api_status' in st.session_state or 'cohere_api_status' in st.session_state):
            if st.button("🔄 Reset verifiche", key="reset_api_status", use_container_width=True):
                if 'github_api_status' in st.session_state:
                    del st.session_state.github_api_status
                    del st.session_state.github_api_message
                if 'cohere_api_status' in st.session_state:
                    del st.session_state.cohere_api_status
                    del st.session_state.cohere_api_message
                st.rerun()
        
        # VERIFICA KNOWLEDGE BASE 
        with st.expander("📚 Verifica Knowledge Base", expanded=False):
            if not os.path.exists(KNOWLEDGE_BASE_DIR):
                st.error(f"❌ Directory {KNOWLEDGE_BASE_DIR} non trovata")
            else:
                files = os.listdir(KNOWLEDGE_BASE_DIR)
                
                pdf_files = [f for f in files if f.lower().endswith('.pdf')]
                txt_files = [f for f in files if f.lower().endswith('.txt')]
                other_files = [f for f in files if not (f.lower().endswith('.pdf') or f.lower().endswith('.txt'))]
                
                st.markdown(f"**Totale file**: {len(files)}")
                st.markdown(f"**PDF**: {len(pdf_files)}")
                st.markdown(f"**TXT**: {len(txt_files)}")
                st.markdown(f"**Altri**: {len(other_files)}")
                
                if st.button("🔍 Verifica leggibilità PDF", key="verify_pdf"):
                    for pdf_file in pdf_files:
                        file_path = os.path.join(KNOWLEDGE_BASE_DIR, pdf_file)
                        try:
                            doc = fitz.open(file_path)
                            page_count = len(doc)
                            text_sample = doc[0].get_text()[:100] if page_count > 0 else ""
                            if text_sample.strip():
                                st.success(f"✅ {pdf_file}: Leggibile ({page_count} pagine)")
                            else:
                                st.warning(f"⚠️ {pdf_file}: PDF leggibile ma senza testo ({page_count} pagine)")
                            doc.close()
                        except Exception as e:
                            st.error(f"❌ {pdf_file}: Errore lettura - {str(e)}")
        
        # METRICHE DI SISTEMA 
        with st.expander("📊 Metriche Sistema", expanded=False):
            st.markdown("#### Utilizzo Memoria Python")
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                st.progress(min(memory_mb / 1000, 1.0))  # Scala su 1GB max
                st.code(f"Memoria utilizzata: {memory_mb:.2f} MB")
            except ImportError:
                st.warning("⚠️ Libreria psutil non disponibile. Installa con: pip install psutil")
            except Exception as e:
                st.error(f"❌ Errore misurazione memoria: {str(e)}")
            
            st.markdown("#### Cache Embedding")
            try:
                cache_files = os.listdir(EMBEDDINGS_CACHE_DIR)
                total_size = sum(os.path.getsize(os.path.join(EMBEDDINGS_CACHE_DIR, f)) for f in cache_files)
                size_mb = total_size / (1024 * 1024)
                
                st.markdown(f"**File nella cache**: {len(cache_files)}")
                st.markdown(f"**Dimensione totale**: {size_mb:.2f} MB")
                
                if len(cache_files) > 0:
                    latest_file = max([os.path.join(EMBEDDINGS_CACHE_DIR, f) for f in cache_files], key=os.path.getmtime)
                    mod_time = os.path.getmtime(latest_file)
                    st.markdown(f"**Ultimo aggiornamento**: {datetime.fromtimestamp(mod_time).strftime('%d/%m/%Y %H:%M')}")
                
                if st.button("🗑️ Svuota cache embedding", key="clear_cache_dev"):
                    with st.spinner("Pulizia cache in corso..."):
                        for file in cache_files:
                            os.remove(os.path.join(EMBEDDINGS_CACHE_DIR, file))
                        st.success("✅ Cache svuotata con successo")
                        st.rerun()
            except Exception as e:
                st.error(f"❌ Errore analisi cache: {str(e)}")
            
            # STATISTICHE API
            st.markdown("#### Chiamate API (Sessione corrente)")
            
            if 'api_calls' not in st.session_state:
                st.session_state.api_calls = {
                    'openweather': 0,
                    'cohere': 0,
                    'github': 0,
                }
            
            # AGGIORNAMENTO CHIAMATE API
            st.code(f"""
OpenWeather API: {st.session_state.api_calls['openweather']} chiamate
Cohere API: {st.session_state.api_calls['cohere']} chiamate
GitHub/OpenAI API: {st.session_state.api_calls['github']} chiamate
            """)
        
        # INFORMAZIONI SU COHERE EMBEDDING
        with st.expander("ℹ️ Info su Cohere Embedding", expanded=False):
            st.markdown("""
            **Cohere Embedding Model con Cache Locale**
            
            Questa applicazione usa Cohere per i modelli di embedding, con un sistema di cache locale per minimizzare le chiamate API.
            
            **Vantaggi:**
            - Prestazioni elevate: elaborazione in cloud senza carico sul dispositivo locale
            - Cache locale: gli embedding calcolati vengono memorizzati localmente per riutilizzo
            - Supporto multilingue: ottimizzato per l'italiano e altre lingue
            - Alta qualità: embeddings di qualità professionale
            
            **Configurazione:**
            L'embedding utilizza il modello "embed-multilingual-v3.0" di Cohere che offre il miglior 
            equilibrio tra prestazioni e supporto linguistico per applicazioni come questa.
            
            **Funzionamento della cache:**
            - Quando si richiede un embedding, prima viene cercato nella cache locale
            - Se non presente, viene generato tramite l'API Cohere e poi salvato
            - La cache riduce il consumo dell'API e migliora i tempi di risposta
            """)
        
        # PULSANTE LOGOUT
        if st.button("🔒 Logout", key="logout_button"):
            st.session_state.authenticated = False
            st.rerun()
            
# FUNZIONE PRINCIPALE  
def main():
    """Funzione principale dell'app"""
    
    initialize_session_state()
    
    # CARICAMENTO CSS
    load_css()
    create_header()
    
    # INIZIALIZZAZIONE EMBEDDING COHERE
    if not st.session_state.get('embed_model_configured', False):
        with st.spinner("Configurazione modello di embedding Cohere..."):
            if setup_embedding_model():
                st.session_state.embed_model_configured = True
            else:
                st.error("Impossibile inizializzare Cohere. Verifica la chiave API e riavvia.")
                return
    
    # INIZIALIZZAZIONE SISTEMA RAG
    if st.session_state.get('rag_index') is None:
        with st.spinner("Caricamento della knowledge base e del database vettoriale..."):
            st.session_state.rag_index = load_rag_index()
            # Non mostrare più il messaggio qui
    
    
    create_sidebar()
    
    current_page = st.session_state.get('page', 'home')
    if current_page == 'home':
        create_home_page()
    elif current_page == 'results':
        create_results_page()
    else:
        st.error("Pagina non riconosciuta")
        st.session_state.page = 'home'
        st.rerun()
    
    create_footer()
   
   
if __name__ == "__main__":
    main()
