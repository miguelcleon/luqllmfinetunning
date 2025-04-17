
# inference_app.py

import gradio as gr
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline

# Paths
OV_MODEL_DIR = "ov_model_ir"
MODEL_PATH = "trainingdata/gemma3_luq_model"  # Path to your fine-tuned model
BASE_MODEL = "google/gemma-3-4b-it"  # Base model name

def build_demo():
    # Load tokenizer
    print("Loading tokenizer from base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Load OpenVINO model
    print("Loading OpenVINO model for inference...")
    model = OVModelForCausalLM.from_pretrained(OV_MODEL_DIR)

    # Create a HF pipeline that under‑the‑hood uses OpenVINO
    print("Creating text-generation pipeline...")
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0,                   # 0 = best available (CPU/GPU/NPU)
        return_full_text=True,
        max_new_tokens=512
    )

    # Generation function
    def generate_response(user_prompt, max_length, temperature, top_p):
        formatted = f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
        out = pipe(
            formatted,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )[0]["generated_text"]
        # extract only model’s reply
        parts = out.split("<start_of_turn>model\n")
        return parts[1].split("<end_of_turn>")[0] if len(parts)>1 else out

    # Build Gradio interface
    with gr.Blocks(title="Fine‑tuned Gemma Research Assistant") as demo:
        gr.Markdown("# Fine‑tuned Gemma Research Assistant")
        gr.Markdown("This model has been fine‑tuned on research papers to help answer questions about scientific topics.")

        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(label="Your question or prompt", lines=4)
                submit_btn = gr.Button("Generate Response", variant="primary")
                clear_btn = gr.Button("Clear")

                with gr.Accordion("Advanced Options", open=False):
                    max_length  = gr.Slider(64, 1024, value=512, step=64, label="Maximum Length")
                    temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
                    top_p       = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top‑p")

            with gr.Column():
                output_text = gr.Textbox(label="Generated Response", lines=20)

        submit_btn.click(generate_response, inputs=[input_text, max_length, temperature, top_p], outputs=output_text)
        clear_btn.click(lambda: ("", ""), inputs=None, outputs=[input_text, output_text])

        gr.Examples(
            examples=[
                ["Summarize the key findings about biodiversity in tropical forests."],
                ["What methods are used to measure climate change effects on ecosystems?"],
                ["Explain the relationship between soil nutrients and plant growth in tropical environments."],
                ["What are the major challenges in conservation of rainforest ecosystems?"]
            ],
            inputs=input_text
        )

    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=True)  # set share=False to disable public link
