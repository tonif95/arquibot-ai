import os
import glob
from dotenv import load_dotenv

# --- IMPORTACIONES DE CARGADORES ---
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

load_dotenv()

DB_CONNECTION = f"postgresql+psycopg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/{os.getenv('POSTGRES_DB')}"

def ingest_data():
    print("🚀 Iniciando ingesta MULTI-FORMATO...")

    # 1. DEFINIR RUTA DE LA CARPETA
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, "knowledge")

    if not os.path.exists(folder_path):
        print(f"❌ Error: No encuentro la carpeta en {folder_path}")
        return

    # 2. DEFINIR CÓMO LEER CADA TIPO DE ARCHIVO
    # Clave: Extensión del archivo | Valor: La clase del Loader
    loader_mapping = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader,
        ".csv": CSVLoader,
    }

    documents = []

    print(f"📂 Escaneando carpeta: {folder_path} ...")

    # 3. ITERAR Y CARGAR
    # DirectoryLoader por defecto es algo básico, a veces es mejor iterar manualmente
    # para tener control total sobre qué loader usa cada archivo.
    for ext, loader_cls in loader_mapping.items():
        # Busca todos los archivos con esa extensión (ej: *.pdf)
        glob_pattern = f"**/*{ext}"
        
        try:
            # Usamos DirectoryLoader filtrando por extensión (glob)
            loader = DirectoryLoader(
                path=folder_path,
                glob=glob_pattern,
                loader_cls=loader_cls,
                show_progress=True,    # Muestra barra de carga
                use_multithreading=True
            )
            
            loaded_docs = loader.load()
            if loaded_docs:
                print(f"   -> Encontrados {len(loaded_docs)} archivos {ext}")
                documents.extend(loaded_docs)
                
        except Exception as e:
            print(f"⚠️ Error cargando archivos {ext}: {e}")

    if not documents:
        print("❌ No se cargaron documentos. Verifica que haya archivos en la carpeta.")
        return

    print(f"📄 Total documentos cargados: {len(documents)}")

    # 4. CHUNKING (Igual que antes)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    print(f"✂️ Texto dividido en {len(splits)} fragmentos.")

    # 5. EMBEDDINGS Y GUARDADO (Igual que antes)
    print("🧠 Cargando modelo de embeddings local...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("💾 Guardando vectores en PostgreSQL...")
    PGVector.from_documents(
        embedding=embeddings,
        documents=splits,
        collection_name="precios_construccion",
        connection=DB_CONNECTION,
        use_jsonb=True,
        pre_delete_collection=True, 
    )

    print("✅ ¡Éxito! Base de conocimientos actualizada.")

if __name__ == "__main__":
    ingest_data()