import re  # Import for cleaning text
from groq import Groq
import pandas as pd

# Configure the Groq API with your API key
client = Groq(api_key="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")  # Replace with your Groq API key

dataset = pd.read_csv(r"C:\Users\soham\Downloads\synthetic_sales_data.csv")

def Groq_Input(user_input):
    prompt = (
        f"genereate response on {user_input}.\n")


    completion = client.chat.completions.create(
        model="deepseek-r1-distill-qwen-32b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=4096,
        top_p=0.95,
        stream=False,
        stop=None,
    )


    generated_code = completion.choices[0].message.content

    print(generated_code)


Groq_Input("what is metaverse?")
