import os
import sys
import tiktoken
import torch
import chainlit

from utils import GPTModel, generate, text_to_token_ids, token_ids_to_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():

    GPT_CONFIG_355M = {
        "vocab_size": 50257, 
        "context_length": 1024,  
        "emb_dim": 1024,   
        "n_heads": 16,  
        "n_layers": 24,       
        "drop_rate": 0.0,     
        "qkv_bias": True    
    }

    tokenizer = tiktoken.get_encoding("gpt2")

    model_path = "gpt2-medium355M-sft.pth"
    if not os.path.exists(model_path):
        print(f"Could not find the {model_path} file.")
        sys.exit()

    checkpoint = torch.load(model_path, weights_only=True)
    model = GPTModel(GPT_CONFIG_355M)
    model.load_state_dict(checkpoint)
    model.to(device)

    return tokenizer, model, GPT_CONFIG_355M


def extract_response(response_text, input_text):
    return response_text[len(input_text):].replace("### Response:", "").strip()


tokenizer, model, model_config = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    """The main Chainlit function."""

    torch.manual_seed(123)

    prompt = f"""Below is an instruction that describes a task. Write a response
    that appropriately completes the request.

    ### Instruction:
    {message.content}
    """

    token_ids = generate( 
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),  
        max_new_tokens=35,
        context_size=model_config["context_length"],
        eos_id=50256
    )

    text = token_ids_to_text(token_ids, tokenizer)
    response = extract_response(text, prompt)

    await chainlit.Message(content=f"{response}").send()
