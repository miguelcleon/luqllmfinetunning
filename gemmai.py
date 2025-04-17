import gradio as gr
from transformers import pipeline

# Initialize the text generation pipeline
pipe = pipeline(
    "text-generation",
    model="google/gemma-2b-it",
    device_map="auto",  # Automatically use GPU if available
    torch_dtype="bfloat16"  # Efficient precision for supported GPUs
)


# Define a function to generate text based on user input
def generate_text(prompt):
    outputs = pipe(prompt, max_length=8000)
    return outputs[0]['generated_text']


# Create a Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs="text",
    outputs="text",
    title="Gemma-2B-IT Text Generation",
    description="Enter a prompt to generate text using the Gemma-2B-IT model."
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()
