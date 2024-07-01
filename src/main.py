from helper import get_openai_api_key
import openai

get_openai_api_key()

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [
        {"role": "system", "content": "You are an helpful assistant"},
        {"role": "user", "content": prompt}
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content

prompt = f"""
Hello my name is kevin
"""


if __name__ == "__main__":
    response = get_completion(prompt)
    print(response)