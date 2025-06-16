import os
import asyncio
import json
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

load_dotenv()

class Settings:
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION")
    AZURE_OPENAI_DEPLOYMENT_ID: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID", "gpt-4o")

try:
    client = AsyncAzureOpenAI(
        azure_endpoint=Settings.AZURE_OPENAI_ENDPOINT,
        api_version=Settings.AZURE_OPENAI_API_VERSION,
        api_key=Settings.AZURE_OPENAI_API_KEY,
        timeout=60.0,  # Increase timeout to 60 seconds
        max_retries=3  # Add retry logic
    )
    print("OpenAI client initialized successfully.")
except Exception as e:
    print(f"Failed to initialize OpenAI client: {e}")
    raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

async def chat():
    system_prompt = {
        "role": "system",
        "content": "You are a learning intent extractor. Your job is to help users define:\n\n{\n  \"topic\": \"\",\n  \"intent\": \"\"\n}\n\n- \"topic\" = what they want to learn about (brief subject)\n- \"intent\" = the perspective or lens they want to study it from (e.g., emotional journey, leadership qualities, communication with God)\n\nAsk only short, helpful questions if needed to clarify either. Once both fields are clear:\n1. Output the JSON\n2. Then say: \"JSON filled. No more questions required.\"\n\nDon’t ask why they want to learn it. Don't mention JSON during the chat. End immediately after printing the JSON."
    }

    messages = []
    extracted_json = None  # <- store JSON here when it's ready

    while True:
        prompt = input("You: ")
        if prompt.lower() in {"exit", "quit"}:
            print("Closing chat...")
            break

        messages.append({"role": "user", "content": prompt})
        print("Assistant:", end=" ", flush=True)

        try:
            stream = await client.chat.completions.create(
                model=Settings.AZURE_OPENAI_DEPLOYMENT_ID,
                messages=[system_prompt] + messages[-10:],  # Keep last 10 messages for context
                stream=True,
                temperature=0.7,
                timeout=30.0
            )

            assistant_content = ""
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    assistant_content += content

            messages.append({"role": "assistant", "content": assistant_content})

            # Try to extract `json = {...}` into a Python variable
            if "json =" in assistant_content:
                try:
                    json_str = assistant_content.split("json =", 1)[1].strip()
                    # Remove anything after the closing curly brace if there's more text
                    json_str = json_str.split("JSON filled.")[0].strip()
                    # Parse JSON
                    extracted_json = json.loads(json_str)
                    print("\n\nJSON filled. No more questions required.")
                    print("Extracted JSON:", extracted_json)
                    break  # Exit the chat loop
                except Exception as e:
                    print(f"\nFailed to extract JSON: {e}")

        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            print("This might be due to:")
            print("1. Network connectivity issues")
            print("2. Azure OpenAI service unavailable")
            print("3. Invalid endpoint or API key")
            print("4. Request timeout")
            print("\nPlease check your configuration and try again.")
            
            fallback_response = "I'm having trouble connecting right now. Please try again in a moment."
            messages.append({"role": "assistant", "content": fallback_response})
            print(f"Assistant: {fallback_response}")

        print("\n")

if __name__ == "__main__":
    asyncio.run(chat())
