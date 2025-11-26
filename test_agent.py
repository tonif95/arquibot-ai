from agent_graph import get_app 
import uuid

# Obtenemos el grafo y el pool
app, pool = get_app()

# ID de hilo (simula un usuario de Telegram, ej: 12345)
thread_id = "usuario_prueba_1"
config = {"configurable": {"thread_id": thread_id}}

print("🏗️ ARQUIBOT INICIADO (Escribe 'salir' para terminar)")

while True:
    user_input = input("\n👷 Tú: ")
    if user_input.lower() == "salir":
        break
    
    # Enviamos el mensaje al grafo
    for event in app.stream({"messages": [("user", user_input)]}, config=config):
        # Imprimimos lo que va pasando en tiempo real
        for key, value in event.items():
            if key == "agente":
                print("🤖 Bot (Pensando...):", value["messages"][-1].content)
            elif key == "herramientas":
                # Mostramos qué herramienta usó y qué devolvió
                last_msg = value["messages"][-1]
                print(f"🛠️ Herramienta usada: {last_msg.name}")
                print(f"📄 Resultado: {last_msg.content}")

# Cerramos conexión al salir
pool.close()