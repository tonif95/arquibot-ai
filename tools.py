from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURACIÓN RAG (Reutilizamos lo del paso 2) ---
DB_CONNECTION = f"postgresql+psycopg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/{os.getenv('POSTGRES_DB')}"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="precios_construccion",
    connection=DB_CONNECTION,
    use_jsonb=True,
)

# --- HERRAMIENTA 1: RAG (Memoria Interna) ---
@tool
def consultar_base_conocimiento(pregunta: str):
    """Útil para consultar precios internos de materiales, normativas de la empresa y salarios de trabajadores."""
    print(f"📚 Consultando RAG sobre: {pregunta}")
    docs = vector_store.similarity_search(pregunta, k=2)
    return "\n".join([d.page_content for d in docs])

# --- HERRAMIENTA 2: Lógica Interna (Calculadora) ---
@tool
def calcular_costo_mano_obra(tipo_trabajador: str, horas: int, cantidad_personas: int = 1):
    """Calcula el costo total de mano de obra dado el tipo, horas y cantidad."""
    # Tarifas base (esto podría venir de una DB también)
    tarifas = {"oficial de primera": 25, "peon": 15, "capataz": 35}
    tipo_normalizado = tipo_trabajador.lower().strip().replace('ó', 'o')
    
    tarifa = tarifas.get(tipo_normalizado,20) # 20 por defecto
    total = tarifa * horas * cantidad_personas
    return f"El costo estimado para {cantidad_personas} {tipo_trabajador}(s) por {horas} horas es de ${total} EUR."

# --- HERRAMIENTA 3: API Externa (Simulada - Clima) ---
@tool
def consultar_clima_obra(ubicacion: str):
    """Consulta el pronóstico del tiempo para saber si se puede trabajar en exteriores."""
    # Aquí iría la llamada a OpenWeatherMap
    return f"Pronóstico para {ubicacion}: Soleado, 22°C. Viento leve. Condiciones perfectas para hormigonado."

# --- HERRAMIENTA 4: API Externa (Simulada - Distancia) ---
@tool
def calcular_logistica_entrega(origen: str, destino: str):
    """Calcula la distancia y tiempo estimado de transporte entre almacén y obra."""
    # Aquí iría la llamada a Google Maps API
    return f"Distancia: 45km. Tiempo estimado con tráfico: 55 minutos."

# --- HERRAMIENTA 5: Acción Crítica (Human-in-the-Loop) ---
@tool
def generar_orden_compra(material: str, cantidad: int, costo_total: float):
    """
    Genera una orden de compra formal. 
    ATENCIÓN: Esta herramienta requiere aprobación humana antes de ejecutarse.
    """
    return f"ORDEN GENERADA: Compra de {cantidad} unidades de {material} por un total de ${costo_total}. ID Transacción: #99281."