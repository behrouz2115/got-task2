import fire
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def generate_text(prompt, length=100, temperature=0.7, top_k=40):
    # Load pre-trained GPT-2 model
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Encode input text as token ids
    encoded_prompt = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text
    output = model.generate(
        encoded_prompt,
        max_length=length,
        temperature=temperature,
        top_k=top_k,
        do_sample=True
    )

    # Decode generated text
    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Print generated text
    print(text)


if __name__ == '__main__':
    fire.Fire(generate_text)
