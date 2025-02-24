from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "google/gemma-2-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")


def generate(prompt):
    # messages = [{"role": "user", "content": prompt}]
    # text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=256)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    return response


generate("정약용에 대해서 알려줘")
generate("정약용에 대해서 알려줘.\n### hint\n告诉我关于郑若镛的事")
