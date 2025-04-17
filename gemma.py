from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="google/gemma-2b-it",
    device_map="auto",          # Auto place on GPU if available
    torch_dtype="bfloat16"      # Efficient precision for supported GPUs
)

# Check device of the model
print(f"Model device: {pipe.model.device}")

# Your prompt
prompt = "Miconia tetrandra is a species within the Melastomataceae family, primarily found in the Caribbean region. This species is notable for its unique floral structure, possessing only four stamens that alternate with the petals—an uncommon trait within the Miconia genus"

# Generate output
outputs = pipe(prompt, max_length=100)

# Print the result
print(outputs[0]['generated_text'])
