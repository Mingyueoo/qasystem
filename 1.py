# Load model directly

from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model():
    # tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
    # model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

    tokenizer = AutoTokenizer.from_pretrained("./CodeLlama-7b-Instruct-hf/")
    model = AutoModelForCausalLM.from_pretrained("./CodeLlama-7b-Instruct-hf/")



    return tokenizer, model


def run_code(tokenizer, model, prompt):
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
    # Answer: """
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
    print(tokenizer.decode(output[0], skip_special_tokens=True))


tokenizer, model = load_model()
prompt = "generate Scatter plot with histograms in Matplotlib"
run_code(tokenizer, model, prompt)
prompt = "generate a function that takes a list of numbers and returns the sum of the list"
run_code(tokenizer, model, prompt)
