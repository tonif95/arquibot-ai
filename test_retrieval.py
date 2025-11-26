import os
from dotenv import load_dotenv
# Cambio aquí también:
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_postgres import PGVector

load_dotenv()

DB_CONNECTION = f"postgresql+psycopg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/{os.getenv('POSTGRES_DB')}"

def test_search(query):
    print(f"\n🔎 Buscando información sobre: '{query}'")

    # Usamos EL MISMO modelo que en la ingesta
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name="precios_construccion",
        connection=DB_CONNECTION,
        use_jsonb=True,
    )

    results = vector_store.similarity_search(query, k=2)

    for i, doc in enumerate(results):
        print(f"--- Resultado {i+1} ---")
        print(doc.page_content)
        print("---------------------")

if __name__ == "__main__":
    test_search("¿Cuál es el precio del hormigón?")
    test_search("¿Es obligatorio usar casco?")