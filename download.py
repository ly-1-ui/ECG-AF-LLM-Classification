from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2-1.5B-Instruct"
save_dir = "./models/qwen2-1.5b"

AutoTokenizer.from_pretrained(model_name).save_pretrained(save_dir)
AutoModelForCausalLM.from_pretrained(model_name).save_pretrained(save_dir)
