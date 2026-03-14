"""最小测试：Llama-70B-INT8 能不能完成一次 forward"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/workspace/models/Llama-3.3-70B-Instruct-INT8"

print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": 0},
    torch_dtype=torch.float16,
    local_files_only=True,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()
print("模型加载完成")

inputs = tokenizer("你好", return_tensors="pt").to(model.device)
print(f"输入 tokens: {inputs['input_ids'].shape}")

with torch.no_grad():
    out = model(**inputs)
print(f"forward 成功，logits shape: {out.logits.shape}")
