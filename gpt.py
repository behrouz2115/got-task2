import tkinter as tk
from tkinter import ttk
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class GPT2GeneratorGUI:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.window = tk.Tk()
        self.window.title("Behrouz Rozmeh")
        self.window.geometry("600x400")

        # Prompt label and entry
        prompt_label = ttk.Label(self.window, text="Prompt:")
        prompt_label.grid(column=0, row=0, padx=10, pady=10)
        self.prompt_entry = ttk.Entry(self.window, width=50)
        self.prompt_entry.grid(column=1, row=0, padx=10, pady=10)

        # Length label and entry
        length_label = ttk.Label(self.window, text="Maximum Length:")
        length_label.grid(column=0, row=1, padx=10, pady=10)
        self.length_entry = ttk.Entry(self.window, width=10)
        self.length_entry.insert(tk.END, "100")
        self.length_entry.grid(column=1, row=1, padx=10, pady=10)

        # Temperature label and entry
        temp_label = ttk.Label(self.window, text="Temperature:")
        temp_label.grid(column=0, row=2, padx=10, pady=10)
        self.temp_entry = ttk.Entry(self.window, width=10)
        self.temp_entry.insert(tk.END, "0.7")
        self.temp_entry.grid(column=1, row=2, padx=10, pady=10)

        # Top-k label and entry
        topk_label = ttk.Label(self.window, text="Top-k:")
        topk_label.grid(column=0, row=3, padx=10, pady=10)
        self.topk_entry = ttk.Entry(self.window, width=10)
        self.topk_entry.insert(tk.END, "40")
        self.topk_entry.grid(column=1, row=3, padx=10, pady=10)

        # Generate button
        generate_button = ttk.Button(
            self.window, text="Generate", command=self.generate_text)
        generate_button.grid(column=1, row=4, padx=10, pady=10)

        # Generated text label and textbox
        text_label = ttk.Label(self.window, text="Generated Text:")
        text_label.grid(column=0, row=5, padx=10, pady=10)
        self.text_box = tk.Text(self.window, height=10, width=50)
        self.text_box.grid(column=1, row=5, padx=10, pady=10)

    def generate_text(self):
        # Get input values
        prompt = self.prompt_entry.get()
        length = int(self.length_entry.get())
        temperature = float(self.temp_entry.get())
        top_k = int(self.topk_entry.get())

        # Encode input text as token ids
        encoded_prompt = self.tokenizer.encode(prompt, return_tensors='pt')

        # Generate text
        output = self.model.generate(
            encoded_prompt,
            max_length=length,
            temperature=temperature,
            top_k=top_k,
            do_sample=True
        )

        # Decode generated text
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Update text box
        self.text_box.delete(1.0, tk.END)
        self.text_box.insert(tk.END, text)

    def run(self):
        self.window.mainloop()
if __name__ == '__main__':
    gui = GPT2GeneratorGUI()
    gui.run()




