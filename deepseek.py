from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("ashad846004/DeepSeek-R1-Medical-COT")
tokenizer = AutoTokenizer.from_pretrained("ashad846004/DeepSeek-R1-Medical-COT")

# Example input
input_text = "A 45-year-old male presents with chest pain and shortness of breath. What is the most likely diagnosis?"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate output
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
