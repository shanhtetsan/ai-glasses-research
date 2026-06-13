from openai import OpenAI
import os

client = OpenAI(
    # If the environment variable is not set, replace with your API key: api_key="sk-xxx"
    api_key="YOUR_DASHSCOPE_API_KEY",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

messages = [{"role": "user", "content": "Who are you?"}]
completion = client.chat.completions.create(
    model="qwen-turbo",  # You can switch to a different deep-thinking model as needed
    messages=messages,
    extra_body={"enable_thinking": True},
    stream=True
)
is_answering = False  # Whether the reply phase has started
print("\n" + "=" * 20 + "Thinking Process" + "=" * 20)
for chunk in completion:
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
        if not is_answering:
            print(delta.reasoning_content, end="", flush=True)
    if hasattr(delta, "content") and delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + "Complete Reply" + "=" * 20)
            is_answering = True
        print(delta.content, end="", flush=True)