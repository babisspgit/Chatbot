from openai import OpenAI
import transformers
from src.prompt import system_instruction


client = OpenAI()

messages = [
    {    "role": "system", "content": system_instruction
    }
    ]

chosen_model0 = "tiiuae/falcon-7b-instruct-v1"
chosen_model1 = "meta-llama/Llama-2-7b-chat-hf"
chosen_model2 = "mistralai/Mistral-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def ask_question(message, model= chosen_model, temperature=0.5):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature  
    )

    return response.choices[0].message.content