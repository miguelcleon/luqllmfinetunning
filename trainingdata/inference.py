
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load the fine-tuned model
model_path = "./trainingdata/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# Create a text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Function to generate text
def generate_text(prompt, max_length=500):
    result = generator(prompt, max_length=max_length, num_return_sequences=1)
    return result[0]['generated_text']

# Interactive loop
print("\nFine-tuned Gemma Model - Test Interface")
print("Enter 'q' or 'quit' to exit")
print("-" * 50)

while True:
    prompt = input("\nEnter a prompt: ")

    if prompt.lower() in ['q', 'quit', 'exit']:
        break

    generated = generate_text(prompt)
    print("\nGenerated text:")
    print("-" * 50)
    print(generated)
    print("-" * 50)
