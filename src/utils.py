import openai

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