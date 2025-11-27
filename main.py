import uvicorn
import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Form
from langchain_core.messages import HumanMessage
from twilio.rest import Client

# Importamos tu grafo
from agent_graph import get_app

# --- 1. Inicialización ---
load_dotenv()

# Credenciales
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_NUMBER") # El número del Sandbox

# Validaciones básicas
if not all([TELEGRAM_TOKEN, TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM]):
    raise ValueError("Faltan variables en el .env (Telegram o Twilio)")

# Cliente Twilio
twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)

# Grafo
try:
    app_graph, pool = get_app()
    print("✅ Grafo y Base de datos conectados.")
except Exception as e:
    print(f"❌ ERROR al iniciar LangGraph: {e}")
    exit()

app = FastAPI(title="ArquiBot API Gateway")

# --- 2. Funciones de Envío (Output) ---

def send_telegram(chat_id: str, text: str):
    """Envía respuesta a Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})

def send_whatsapp(to_number: str, text: str):
    """Envía respuesta a WhatsApp (Twilio Sandbox)"""
    try:
        twilio_client.messages.create(
            from_=TWILIO_FROM,
            body=text,
            to=to_number
        )
    except Exception as e:
        print(f"❌ Error enviando WhatsApp: {e}")

# --- 3. Lógica Central del Agente (Agnóstica del canal) ---

def run_agent(user_input: str, thread_id: str, reply_callback):
    """
    Ejecuta la lógica de LangGraph.
    - user_input: El texto que escribió el usuario.
    - thread_id: ID único (ChatID de Telegram o Teléfono de WhatsApp).
    - reply_callback: Función para responder (send_telegram o send_whatsapp).
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    # A. Revisar si hay interrupciones previas (HITL)
    try:
        state = app_graph.get_state(config)
    except Exception:
        state = None # Chat nuevo

    if state and state.next and state.next[0] == '__interrupt__':
        # El grafo estaba pausado esperando aprobación
        if user_input.lower() == "aprobar orden":
            reply_callback("✅ **ORDEN APROBADA!** Reanudando el proceso...")
            
            # Reanudamos el grafo
            final_output = app_graph.invoke(
                input={"messages": [HumanMessage(content="ORDEN APROBADA")]}, 
                config=config
            )
            bot_response = final_output["messages"][-1].content
            reply_callback(bot_response)
            return

        elif user_input.lower() == "cancelar orden":
            pool.delete_thread(thread_id) # Borrar memoria
            reply_callback("❌ **ORDEN CANCELADA.** Memoria limpiada.")
            return

        else:
            reply_callback("🚨 **APROBACIÓN PENDIENTE.**\nEscribe `APROBAR ORDEN` o `CANCELAR ORDEN`.")
            return

    # B. Flujo Normal (Nuevo mensaje)
    interrupt_on_tools = ["generar_orden_compra"] 
    
    final_output = app_graph.invoke(
        input={"messages": [HumanMessage(content=user_input)]}, 
        config=config,
        interrupt_before=[("herramientas", {"tools": interrupt_on_tools})]
    )

    # C. Verificar si el LLM decidió detenerse
    if app_graph.get_state(config).next:
        reply_callback("🚨 **ACCIÓN CRÍTICA DETECTADA: ORDEN DE COMPRA.**\nEl Agente solicita aprobación.\nResponde: `APROBAR ORDEN` o `CANCELAR ORDEN`.")
    else:
        # Respuesta normal del bot
        bot_response = final_output["messages"][-1].content
        reply_callback(bot_response)


# --- 4. Webhooks (Inputs) ---

@app.post("/webhook/telegram")
async def telegram_webhook(request: Request):
    try:
        data = await request.json()
    except:
        raise HTTPException(400, "Error JSON")

    if "message" not in data: return {"status": "ok"}
    
    chat_id = str(data["message"]["chat"]["id"])
    text = data["message"].get("text", "")
    
    # Usamos una lambda para pasar la función de envío correcta
    run_agent(text, chat_id, lambda msg: send_telegram(chat_id, msg))
    
    return {"status": "ok"}

@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    # Twilio envía Form Data, necesitamos procesarlo así:
    form_data = await request.form()
    
    user_number = form_data.get("From") # Ej: whatsapp:+34600112233
    text = form_data.get("Body", "")    # El mensaje del usuario

    if not user_number: return {"status": "error"}

    print(f"📩 WhatsApp de {user_number}: {text}")

    # Ejecutamos el agente usando el número de teléfono como thread_id
    run_agent(text, user_number, lambda msg: send_whatsapp(user_number, msg))

    return {"status": "ok"}

@app.get("/")
def root():
    return {"status": "ArquiBot is running 🤖"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)