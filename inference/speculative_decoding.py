from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time

model1 = ''
model2 = ''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(model1)

prompt = ''
formatted_prompt = f"### Human: {prompt}### Assitant:"
inputs = tokenizer(formatted_prompt, return_tensors='pt').to(device)
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True)
model.load_adapter(peft_model_id)

model.config.use_cache = True
assistant_model = AutoModelForCausalLM.from_pretrained(model2).half().to(device)
assistant_model.config.use_cache = True

outputs = model.generate(**inputs, assistant_model=assistant_model, max_new_tokens=512)
outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print(outputs)
