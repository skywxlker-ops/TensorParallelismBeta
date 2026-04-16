import os
os.environ["SAFETENSORS_DISABLE_MMAP"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# 1. Configuration - Using Bfloat16 for stability
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-31B-it",
    quantization_config=quant_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    offload_folder="offload_dir"
)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-31B-it")

# 2. Use the Chat Template (Crucial for Instruction models)
chat = [
    {"role": "user", "content": "Write a short poem about a robot learning to paint."},
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 3. Generation - Added repetition_penalty to stop the "LCTCTC" loop
print("\n--- Gemma is thinking... ---")
outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2 # Prevents looping gibberish
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
