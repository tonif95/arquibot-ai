import uvicorn
import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from langchain_core.messages import HumanMessage, AIMessage

# Importamos el grafo y la configuración
from agent_graph import get_app

# --- Inicialización ---
load_dotenv()

# Obtener credenciales
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN no encontrado en .env")

# Obtener el grafo una sola vez al iniciar el servidor
try:
    app_graph, pool = get_app()
except Exception as e:
    print(f"❌ ERROR al iniciar LangGraph/Postgres: {e}")
    exit()

# Inicialización de FastAPI
app = FastAPI(title="ArquiBot API Gateway")

# Función para enviar respuestas de vuelta a Telegram
def send_telegram_message(chat_id: int, text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    print(f"URL de respuesta generada: {url}")
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }
    response = requests.post(url, json=payload)
    if not response.ok:
        print(f"ERROR enviando a Telegram: {response.text}")

# --- 4.2 Lógica del Webhook y LangGraph ---

@app.post("/webhook")
async def handle_webhook(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Cuerpo de petición inválido")

    # Si no es un mensaje (ej. es una edición), ignoramos
    if "message" not in data:
        return {"status": "ok"}

    message = data["message"]
    chat_id = message["chat"]["id"]
    user_input = message.get("text", "")

    # Configuración de LangGraph
    thread_id = str(chat_id)
    config = {"configurable": {"thread_id": thread_id}}
    
    # ----------------------------------------------------
    # 1. Lógica Human-in-the-Loop (HITL)
    # ----------------------------------------------------
    
    # Primero, verificamos si el grafo está pausado por el HITL
    # El grafo se pausó en el nodo 'herramientas' al detectar la herramienta 'generar_orden_compra'
    try:
        state = app_graph.get_state(config)
    except Exception:
        state = None # No hay estado previo, es un nuevo chat
        
    if state and state.next:
        # Si el estado tiene next, significa que el grafo está pausado
        if state.next[0] == '__interrupt__':
            # Aquí implementamos el comando de reanudación
            if user_input.lower() == "aprobar orden":
                # Reanudamos el grafo, simulando una respuesta 'tool_call' vacía para continuar
                send_telegram_message(chat_id, "✅ **ORDEN APROBADA!** Reanudando el proceso de compra...")
                
                # Necesitamos un placeholder para reanudar el estado
                # En la lógica real, pasarías la respuesta de aprobación al nodo
                
                # Reanudamos el grafo. Necesita el mensaje del humano (tú)
                final_output = app_graph.invoke(
                    # El mensaje que reanuda el grafo
                    input={"messages": [HumanMessage(content="ORDEN APROBADA")]}, 
                    config=config
                )
                
                # Buscamos el último mensaje del bot que debería ser el resultado de la herramienta
                bot_response = final_output["messages"][-1].content
                send_telegram_message(chat_id, bot_response)
                return {"status": "ok"}
                
            elif user_input.lower() == "cancelar orden":
                # Si se cancela, simplemente terminamos la conversación y limpiamos
                pool.delete_thread(thread_id) # Borramos el estado
                send_telegram_message(chat_id, "❌ **ORDEN CANCELADA.** El proceso ha sido terminado y la memoria limpiada.")
                return {"status": "ok"}

            else:
                send_telegram_message(chat_id, 
                    "🚨 **ATENCIÓN: Proceso de APROBACIÓN PENDIENTE.**\n"
                    "Por favor, escribe:\n"
                    "- `APROBAR ORDEN` para continuar con la compra.\n"
                    "- `CANCELAR ORDEN` para terminar la solicitud.")
                return {"status": "ok"}


    # ----------------------------------------------------
    # 2. Flujo Normal de Conversación
    # ----------------------------------------------------
    
    # Configuramos la interrupción aquí (antes de que el agente piense)
    # Si el LLM decide usar la herramienta 'generar_orden_compra', el grafo se detiene
    interrupt_on_tools = ["generar_orden_compra"] 
    
    # El LLM ve el mensaje y decide la acción
    final_output = app_graph.invoke(
        input={"messages": [HumanMessage(content=user_input)]}, 
        config=config,
        # La clave para el HITL
        interrupt_before=[("herramientas", {"tools": interrupt_on_tools})]
    )

    # El HITL no ha parado la ejecución, procesamos el resultado
    if app_graph.get_state(config).next:
        # Si hay un 'next' después de la invocación, significa que se detuvo.
        # Esto ocurre cuando el LLM decide usar la herramienta de compra.
        send_telegram_message(chat_id, 
            "🚨 **ACCIÓN CRÍTICA DETECTADA: ORDEN DE COMPRA.**\n"
            "El Agente ArquiBot ha decidido usar la función de compra. Se requiere su aprobación.\n"
            "Por favor, envíe `APROBAR ORDEN` o `CANCELAR ORDEN`."
        )
    else:
        # No se detuvo, solo respondió normalmente
        bot_response = final_output["messages"][-1].content
        send_telegram_message(chat_id, bot_response)

    return {"status": "ok"}

# Endpoint para verificar que el servidor está vivo
@app.get("/")
def read_root():
    return {"status": "ArquiBot is running"}

# --- 4.3 Inicio del Servidor ---

if __name__ == "__main__":
    print("Servidor FastAPI iniciado. Conectándose a Postgres...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)