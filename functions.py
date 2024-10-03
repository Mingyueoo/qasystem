from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def load_model():
    # load the model from HF
    # tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
    # generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
    # config = PeftConfig.from_pretrained("mingyue0101/super-cool-instruct")
    # model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
    # model = PeftModel.from_pretrained(model, "mingyue0101/super-cool-instruct")

    # load the model from local directory
    # tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
    # generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
    # config = PeftConfig.from_pretrained("./codellama_Ming05/")
    # model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
    # model = PeftModel.from_pretrained(model, "./codellama_Ming05/")

    # Load the model and tokenizer directly from the local directory path
    tokenizer = AutoTokenizer.from_pretrained("./CodeLlama-7b-Instruct-hf/", local_files_only=True)
    config = PeftConfig.from_pretrained("./codellama_Ming05/", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained("./CodeLlama-7b-Instruct-hf/", local_files_only=True)
    model = PeftModel.from_pretrained(model, "./codellama_Ming05/", local_files_only=True)
    return tokenizer,model

def run_code_generator(prompt,tokenizer,model):
    generation_kwargs = {
        "do_sample": False,  # set to true if temperature is not 0
        "temperature": None,
        "max_new_tokens": 1024,
        "top_k": 50,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
    }

    generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    prompt_template = "### USER:\n{human_prompt}\n\n### RESPONSE:\n"
    full_prompt = prompt_template.format(human_prompt=prompt)


    tokens = tokenizer(full_prompt, return_tensors="pt")
    # Remove unneeded kwargs
    if generation_kwargs["do_sample"] == False:
        generation_kwargs.pop("temperature")
        generation_kwargs.pop("top_k")
        generation_kwargs.pop("top_p")
    output = model.generate(
        tokens["input_ids"],
        **generation_kwargs,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


# sample function to test
# def run_code_generator(prompt):
#     return "I'm example code " * 10
