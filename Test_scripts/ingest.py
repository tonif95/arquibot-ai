import os
from dotenv import load_dotenv

# Nuevas importaciones para usar modelo local
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # <--- CAMBIO AQUÍ
from langchain_postgres import PGVector

load_dotenv()

# Conexión DB
DB_CONNECTION = f"postgresql+psycopg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/{os.getenv('POSTGRES_DB')}"

def ingest_data():
    print("🚀 Iniciando ingesta de datos con Embeddings Locales...")

    # 1. CARGAR
    file_path = "../knowledge/datos_constructora_2025.txt"
    if not os.path.exists(file_path):
        print(f"❌ Error: No encuentro el archivo en {file_path}")
        return

    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print(f"📄 Documento cargado.")

    # 2. CHUNKING
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    print(f"✂️ Texto dividido en {len(splits)} fragmentos.")

    # 3. EMBEDDINGS (CAMBIO IMPORTANTE)
    # En lugar de llamar a Google, descargamos un modelo pequeño (aprox 80MB) la primera vez.
    print("🧠 Cargando modelo de embeddings local (HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("💾 Guardando vectores en PostgreSQL...")
    
    # ATENCIÓN: Usamos pre_delete_collection=True para borrar lo anterior y evitar errores de dimensiones
    PGVector.from_documents(
        embedding=embeddings,
        documents=splits,
        collection_name="precios_construccion",
        connection=DB_CONNECTION,
        use_jsonb=True,
        pre_delete_collection=True, 
    )

    print("✅ ¡Éxito! Datos guardados sin usar cuota de API.")

if __name__ == "__main__":
    ingest_data()