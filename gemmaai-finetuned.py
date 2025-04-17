import gradio as gr
from optimum.intel.openvino import OVModelForCausalLM
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import os

# Define paths
OV_MODEL_DIR = "ov_model_ir"
MODEL_PATH = "trainingdata/gemma3_luq_model"  # Path to your fine-tuned model
BASE_MODEL = "google/gemma-3-4b-it"  # Base model name

# Check for GPU availability
use_gpu = torch.cuda.is_available()
device = "cuda" if use_gpu else "cpu"
print(f"Using device: {device}")

# Set up quantization if using GPU
if use_gpu:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )
else:
    bnb_config = None

# Load base model and tokenizer
print(f"Loading base model: {BASE_MODEL}")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto" if use_gpu else "cpu",
    torch_dtype=torch.float16 if use_gpu else torch.float32,
    quantization_config=bnb_config if use_gpu else None
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# load the OpenVINO model (this will dispatch to CPU/GPU/NPU automatically)
model = OVModelForCausalLM.from_pretrained(OV_MODEL_DIR)

# Load LoRA weights
print(f"Loading LoRA weights from: {MODEL_PATH}")
model = PeftModel.from_pretrained(model, MODEL_PATH)

# Create a pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,
    return_full_text=True,
    max_new_tokens=1024
)


# Function to generate text with the fine-tuned model
def generate_response(user_prompt, max_length=512, temperature=0.7, top_p=0.9):
    # Format input in the instruction format for Gemma
    formatted_prompt = f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"

    # Generate response
    outputs = pipe(
        formatted_prompt,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = outputs[0]['generated_text']

    # Extract just the model's response
    response_parts = generated_text.split("<start_of_turn>model\n")
    if len(response_parts) > 1:
        model_response = response_parts[1].split("<end_of_turn>")[0] if "<end_of_turn>" in response_parts[1] else \
        response_parts[1]
        return model_response
    else:
        return generated_text


# Define the Gradio interface
with gr.Blocks(title="Fine-tuned Gemma Research Assistant") as demo:
    gr.Markdown("# Fine-tuned Gemma Research Assistant")
    gr.Markdown("This model has been fine-tuned on research papers to help answer questions about scientific topics.")

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Your question or prompt",
                placeholder="Ask a question about research papers...",
                lines=4
            )

            with gr.Row():
                submit_btn = gr.Button("Generate Response", variant="primary")
                clear_btn = gr.Button("Clear")

            with gr.Accordion("Advanced Options", open=False):
                max_length = gr.Slider(
                    minimum=64, maximum=1024, value=512, step=64,
                    label="Maximum Length"
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                    label="Temperature (higher = more creative)"
                )
                top_p = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                    label="Top-p (nucleus sampling)"
                )

        with gr.Column():
            output_text = gr.Textbox(
                label="Generated Response",
                lines=20
            )

    # Handle button actions
    submit_btn.click(
        fn=generate_response,
        inputs=[input_text, max_length, temperature, top_p],
        outputs=output_text
    )
    clear_btn.click(
        fn=lambda: ("", None),
        inputs=None,
        outputs=[input_text, output_text]
    )

    # Example prompts
    gr.Examples(
        examples=[
            ["Summarize the key findings about biodiversity in tropical forests."],
            ["What methods are used to measure climate change effects on ecosystems?"],
            ["Explain the relationship between soil nutrients and plant growth in tropical environments."],
            ["What are the major challenges in conservation of rainforest ecosystems?"]
        ],
        inputs=input_text
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True)  # Set share=False if you don't want a public link