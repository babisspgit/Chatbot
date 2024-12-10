import chainlit as cl
from src.llm import ask_question, messages


@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...

    messages.append({"role": "user", "content": message.content})
    response = ask_question(messages)
    messages.append({"role": "system", "content": response})

    # Send a response back to the user
    await cl.Message(
        #content=f"Received: {message.content}",
        content=response,

    ).send()
