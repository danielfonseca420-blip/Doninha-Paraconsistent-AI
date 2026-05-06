import chainlit as cl
import ollama

# ────────────────────────────────────────────────
# Configurações fixas (mude aqui o que precisar)
# ────────────────────────────────────────────────
MODEL_NAME = "IA-Doninha-llama2:latest"          # ← coloque o nome exato do seu modelo
SYSTEM_PROMPT = """
Você é um assistente extremamente útil, direto e sarcástico quando faz sentido.
Responda em português do Brasil, de forma clara e concisa.
"""  # personalize bastante aqui!

# ────────────────────────────────────────────────

@cl.on_chat_start
async def start():
    # Mensagem de boas-vindas que aparece quando o usuário entra
    await cl.Message(
        content="Bem Vindo à Inteligência Artificial da Operação Doninha! Não faça perguntas idiotas"
    ).send()

    # Opcional: mostra um "pensando..." enquanto carrega
    cl.user_session.set("history", [])


@cl.on_message
async def main(message: cl.Message):
    # Pega o histórico da sessão (para manter contexto)
    history = cl.user_session.get("history") or []

    # Adiciona a mensagem do usuário no histórico
    history.append({"role": "user", "content": message.content})

    # Mostra "pensando..." na interface
    msg = cl.Message(content="")
    await msg.send()

    # Chama o Ollama com streaming (resposta aparece letra por letra)
    try:
        stream = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *history
            ],
            stream=True,
            options={
                "temperature": 0.7,
                "num_ctx": 8192  # aumenta se seu modelo suportar mais contexto
            }
        )

        full_response = ""

        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                token = chunk["message"]["content"]
                full_response += token
                await msg.stream_token(token)

        # Finaliza a mensagem
        await msg.update()

        # Salva a resposta da IA no histórico
        history.append({"role": "assistant", "content": full_response})
        cl.user_session.set("history", history)

    except Exception as e:
        await cl.Message(
            content=f"Ops... deu ruim aqui: {str(e)}\nTenta de novo?"
        ).send()


# Opcional: botão para limpar conversa
@cl.action_callback(name="Limpar conversa")
async def clear_conversation():
    cl.user_session.set("history", [])
    await cl.Message(content="Conversa zerada! Pode começar do zero.").send()