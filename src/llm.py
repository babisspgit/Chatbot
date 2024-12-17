from openai import OpenAI
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.prompt import system_instruction
from langchain.llms import HuggingFacePipeline
#from langchain_huggingface.llms import HuggingFacePipeline


# Initialize Hugging Face Model
def initialize_model():
    model_name = "tiiuae/falcon-7b-instruct"  # Replace with your desired Hugging Face model

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    # Create a text-generation pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=500, temperature=0.7)

    # Wrap in LangChain's HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# Initialize the model
llm = initialize_model()

# Test the model
response = llm("Explain the concept of AI in simple terms.")
print(response)



client = OpenAI()
messages = [
    {    "role": "system", "content": system_instruction
    }
    ]

model0 = "tiiuae/falcon-7b-instruct-v1"
model1 = "meta-llama/Llama-2-7b-chat-hf"
model2 = "mistralai/Mistral-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def ask_question(message, model= model0, temperature=0.5):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature  
    )

    return response.choices[0].message.content