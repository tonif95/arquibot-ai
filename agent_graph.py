import os
from dotenv import load_dotenv

# LangGraph y LangChain
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage

# Importamos nuestras herramientas
from tools import (
    consultar_base_conocimiento,
    calcular_costo_mano_obra,
    consultar_clima_obra,
    calcular_logistica_entrega,
    generar_orden_compra,
    send_email
)
# Importamos pool de conexión (necesario para el checkpointer)
from psycopg_pool import ConnectionPool

load_dotenv()

# 1. Definir el Estado del Agente
# El estado es simplemente una lista de mensajes que va creciendo
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Configurar el Modelo (Gemini) y las Herramientas
lista_herramientas = [
    consultar_base_conocimiento,
    calcular_costo_mano_obra,
    consultar_clima_obra,
    calcular_logistica_entrega,
    generar_orden_compra,
    send_email
]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_con_herramientas = llm.bind_tools(lista_herramientas)

# 3. Definir el Nodo del Agente (El Cerebro)
def nodo_agente(state: AgentState):
    # Mensaje de sistema para darle personalidad
    sys_msg = SystemMessage(content="""Eres ArquiBot, un asistente experto en gestión de construcción. 
    Usa tus herramientas para responder. 
    Para preguntas sobre precios, costos o normas, **SIEMPRE DEBES CONSULTAR TU BASE DE CONOCIMIENTO (consultar_base_conocimiento)**.
    Si el usuario pregunta algo relacionado con cuanto le va a costar la mano de obra, tienes que usar la herramienta calcular_costo_mano_obra.
    Si vas a comprar algo, **TU PROCESO DEBE SER ESTRICTAMENTE**:
    1. Usar la tool **generar_orden_compra**.
    2. **INMEDIATAMENTE DESPUÉS** de generar la orden, debes usar la tool **send_email** para enviar la orden al proveedor. 
                            La orden generada debe ser el contenido principal del correo y el asunto del correo debe ser orden de compra.                              
    """)
    
    return {"messages": [llm_con_herramientas.invoke([sys_msg] + state["messages"])]}

# 4. Construir el Grafo
builder = StateGraph(AgentState)

builder.add_node("agente", nodo_agente)
builder.add_node("herramientas", ToolNode(lista_herramientas))

builder.add_edge(START, "agente")
# Lógica condicional: Si el LLM quiere usar una herramienta, va a "herramientas", si no, termina.
builder.add_conditional_edges("agente", tools_condition, {"tools": "herramientas", END: END})
builder.add_edge("herramientas", "agente")

# 5. Configurar Persistencia (PostgreSQL Checkpointer)
# Esto permite que si reinicias el bot, recuerde la conversación
DB_URI = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/{os.getenv('POSTGRES_DB')}"

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

# --- FUNCIÓN FACTORY ---
# Envolvemos la creación del grafo en una función para gestionar la conexión a DB
def get_app():
    pool = ConnectionPool(conninfo=DB_URI, kwargs=connection_kwargs)
    checkpointer = PostgresSaver(pool)
    
    # IMPORTANTE: Configuramos las tablas de checkpoints la primera vez
    checkpointer.setup()
    
    # Aquí definimos el HUMAN-IN-THE-LOOP
    # Le decimos: "Párate antes de ejecutar la herramienta 'generar_orden_compra'"
    # Nota: LangGraph detecta la herramienta dentro del nodo 'herramientas'
    
    # Compilamos el grafo con memoria e interrupción
    app = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=[], # Lo configuraremos dinámicamente o lo dejaremos manual para el paso 4
    )
    return app, pool