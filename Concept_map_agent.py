import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# Load Azure OpenAI config from environment variables or hardcode here
AZURE_OPENAI_KEY = "2be1544b3dc14327b60a870fe8b94f35"
AZURE_OPENAI_ENDPOINT = "https://notedai.openai.azure.com"
AZURE_OPENAI_VERSION = "2024-06-01"
AZURE_DEPLOYMENT_NAME = "gpt-4o"

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

def create_concept_map(topic):
    prompt = f"""
You are an expert concept map creator, the best there is.

Create a highly detailed and specific concept map for the topic '{topic}'.

Include:
- Key concepts that must appear in explanations.
- Important keywords to look for.
- Words or topics to exclude (especially non A-Level topics like CBSE, 9th grade, school level).
- Relevant formulas or principles.
- Common misconceptions if applicable.
- Notes about the topic if relevant.

Return the concept map as a valid JSON object with the following keys:

- key_concepts: A list of objects with 'name', 'definition', and optionally 'sub_concepts'.
- include_keywords: List of exact keywords or phrases that are relevant.
- exclude_keywords: List of keywords or phrases to filter out irrelevant content.
- formulas: List of relevant formulas as strings.
- common_misconceptions: List of common misconceptions (optional).
- notes: Additional notes or instructions (optional).

Only output the JSON object, nothing else. The JSON should be properly formatted.

Also dont make the json too long let it be short but have all the points
"""

    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.0,
    )

    content = response.choices[0].message.content.strip()

    # Print raw GPT output for debugging
    print("RAW GPT OUTPUT:\n", content)

    # Clean up triple backticks if present
    if content.startswith("```json"):
        content = content[7:-3].strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()

    # Attempt to parse JSON
    try:
        concept_map = json.loads(content)
        return concept_map
    except json.JSONDecodeError as e:
        print(f"Failed to parse concept map JSON:\n{content}")
        raise e



if __name__ == "__main__":
    topic = "Possion Distribution A-levels"
    concept_map = create_concept_map(topic)
    print(json.dumps(concept_map, indent=2))
