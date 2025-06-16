import requests
import time
import json
import os
import asyncio
from urllib.parse import urlencode
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

#Quary and website count
q=10
websites=5

# === Configuration ===
class Settings:
    # Google Search API
    GOOGLE_API_KEY = "AIzaSyCk4DYKCm5sSLz63aFUlVk8E04QPSvjXT8"
    GOOGLE_CX = "53459b243c2c34e0c"
    
    # Azure OpenAI
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "2be1544b3dc14327b60a870fe8b94f35")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://notedai.openai.azure.com")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    AZURE_OPENAI_DEPLOYMENT_ID = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID", "gpt-4o")

# === Global Variables ===
total_prompt_tokens = 0
total_completion_tokens = 0

# Initialize components
model = SentenceTransformer("all-MiniLM-L6-v2")
ENCODING = tiktoken.encoding_for_model("gpt-4o")

# Azure OpenAI Client
try:
    client = AsyncAzureOpenAI(
        azure_endpoint=Settings.AZURE_OPENAI_ENDPOINT,
        api_version=Settings.AZURE_OPENAI_API_VERSION,
        api_key=Settings.AZURE_OPENAI_API_KEY,
        timeout=60.0,
        max_retries=3
    )
    print("✅ OpenAI client initialized successfully.")
except Exception as e:
    print(f"❌ Failed to initialize OpenAI client: {e}")
    raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

# Headers for direct API calls
headers = {
    "Content-Type": "application/json",
    "api-key": Settings.AZURE_OPENAI_API_KEY
}

# === Utility Functions ===
def count_tokens_from_messages(messages, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    tokens_per_message = 3
    tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def azure_chat_completion(messages):
    global total_prompt_tokens, total_completion_tokens

    url = f"{Settings.AZURE_OPENAI_ENDPOINT}/openai/deployments/{Settings.AZURE_OPENAI_DEPLOYMENT_ID}/chat/completions?api-version={Settings.AZURE_OPENAI_API_VERSION}"

    prompt_tokens = count_tokens_from_messages(messages, model="gpt-4o")

    response = requests.post(url, headers=headers, json={
        "messages": messages,
        "temperature": 0.7
    })
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    completion_tokens = len(ENCODING.encode(content))

    total_prompt_tokens += prompt_tokens
    total_completion_tokens += completion_tokens

    print(f"📏 Tokens - Prompt: {prompt_tokens} | Completion: {completion_tokens} | Total: {prompt_tokens + completion_tokens}")
    return content

# === Chat Function to Extract Intent ===
async def extract_intent_chat():
    """Interactive chat to extract topic and intent from user"""
    print("\n" + "="*60)
    print("🤖 AI RESEARCH ASSISTANT - INTENT EXTRACTION")
    print("="*60)
    print("Let's define what you want to research and from what perspective.")
    print("Type 'exit' or 'quit' to stop.\n")

    system_prompt = {
  "role": "system",
  "content": """You are a learning intent extractor. Your job is to define:

{
  "topic": "",
  "intent": ""
}

Definitions:
- "topic" = the subject they want to learn (e.g., "keyboard", "Moses", "statistics")
- "intent" = the learner's starting point (how much they know), and the *approach or perspective* they want to take (e.g., emotional journey, technical mastery, performance growth, worship, etc.)

Your job is to **ask short, focused questions** until BOTH are clear.

✅ When topic is vague or ambiguous, clarify:  
- “Do you mean building a keyboard or learning to play one?”

✅ When intent is vague or shallow, ask:  
1. “Have you learned or tried this before? What’s your current level?”  
2. "How do you want to learn it ? the *approach or perspective* they want to take

✅ Your goal is a clear picture of **where they are now and how they want to go deeper.**

Once both are clear:
1. Output the JSON in this exact format: {"topic": "...", "intent": "..."}
2. Then say: "JSON filled. Starting research..."

Do not ask why they want to learn. End immediately after the JSON.
"""
}



    messages = []
    extracted_json = None

    while True:
        try:
            prompt = input("You: ")
            if prompt.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                return None

            messages.append({"role": "user", "content": prompt})
            print("Assistant: ", end="", flush=True)

            stream = await client.chat.completions.create(
                model=Settings.AZURE_OPENAI_DEPLOYMENT_ID,
                messages=[system_prompt] + messages[-10:],
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

            # Extract JSON from response
            if "{" in assistant_content and "}" in assistant_content:
                try:
                    # Find JSON in the response
                    start = assistant_content.find("{")
                    end = assistant_content.find("}", start) + 1
                    json_str = assistant_content[start:end]
                    extracted_json = json.loads(json_str)
                    
                    if "topic" in extracted_json and "intent" in extracted_json:
                        print(f"\n\n✅ Extracted: {extracted_json}")
                        return extracted_json
                        
                except json.JSONDecodeError:
                    pass  # Continue chat if JSON parsing fails

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again.")

        print("\n")

# === Research Functions ===
def generate_intent_based_query(topic, intent):
    """Generate a search query that combines topic with intent perspective"""
    print(f"🎯 Generating intent-based search query...")
    messages = [
        {
            "role": "system",
            "content": "You are a search query specialist. Create a focused Google search query that combines the topic with the user's specific intent or perspective."
        },
        {
            "role": "user",
            "content": f"""
Topic: "{topic}"
User's Intent/Perspective: "{intent}"

Create a single, focused Google search query (3-8 words) that will find results specifically related to the user's intent about this topic. 

Examples:
- Topic: "Moses in the Bible", Intent: "leadership qualities" → "Moses leadership qualities biblical"
- Topic: "Climate change", Intent: "economic impact" → "climate change economic impact costs"

Return only the search query, nothing else.
"""
        }
    ]
    
    query = azure_chat_completion(messages).strip().strip('"')
    print(f"🔍 Generated query: '{query}'")
    return query

def google_search(query, num_results=websites):
    """Search Google and return URLs"""
    print(f"🔍 Searching Google: '{query}'")
    query_params = urlencode({"q": query, "cx": Settings.GOOGLE_CX, "key": Settings.GOOGLE_API_KEY, "num": num_results})
    url = f"https://www.googleapis.com/customsearch/v1?{query_params}"
    
    response = requests.get(url)
    if response.status_code == 429:
        print("⚠️ Rate limited, waiting 2 seconds...")
        time.sleep(2)
        response = requests.get(url)
    
    data = response.json()
    time.sleep(0.2)
    results = [item["link"] for item in data.get("items", [])][:num_results]
    print(f"✅ Found {len(results)} results for '{query}'")
    return results

def scrape_page_text(url):
    """Scrape text content from a webpage"""
    try:
        print(f"🌐 Scraping: {url[:60]}...")
        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        print(f"❌ Failed to scrape {url}: {str(e)}")
        return ""

def summarize_pages(urls, topic, intent):
    """Summarize scraped pages with intent focus"""
    print(f"📚 Summarizing {len(urls)} pages with intent focus...")
    contents = []
    for url in urls:
        text = scrape_page_text(url)
        if text:
            contents.append(text)
    
    combined_text = "\n\n".join(contents)
    
    if not combined_text:
        print("⚠️ No content found for summarization")
        return "", 0

    prompt = [
        {
            "role": "system",
            "content": "You are a research assistant that summarizes information with laser focus on the user's specific intent and perspective."
        },
        {
            "role": "user",
            "content": f"""
Topic: "{topic}"  
User's Intent/Perspective: "{intent}"

Summarize the following content, but ONLY focus on information that directly relates to the user's intent. Ignore general information that doesn't serve their specific perspective.

Key instructions:
- Extract only content that addresses the user's intent
- Organize insights around the user's perspective
- Skip generic or unrelated information
- Be specific and detailed about relevant aspects

Content to analyze:
{combined_text}
"""
        }
    ]
    
    summary = azure_chat_completion(prompt)
    print(f"✅ Intent-focused summary completed ({len(summary)} chars)")
    return summary, len(contents)

def refine_queries(summary, topic, intent):
    """Generate refined search queries based on summary and intent"""
    print("🧠 Generating refined queries based on intent...")
    messages = [
        {
            "role": "system",
            "content": "You are a research specialist who creates targeted search queries that dig deeper into specific aspects related to the user's intent."
        },
        {
            "role": "user",
            "content": f"""
Topic: "{topic}"  
User's Intent: "{intent}"

Based on this summary, generate 7 specific search queries that will find MORE information specifically about the USER'S INTENT. Each query should:
- Target a different aspect of the user's intent (VERY IMPORTENT)
- Be specific enough to find focused results
- Use 3-6 words maximum
- Avoid generic terms

Current summary:
{summary}

Format: Return only the search queries, one per line, no numbering or bullets.
"""
        }
    ]

    raw_queries = azure_chat_completion(messages)
    queries = [q.strip("-• ").strip() for q in raw_queries.split("\n") if q.strip() and len(q.strip()) > 5]
    print("🔍 Refined intent-based queries:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    return queries

def process_query(query, topic, intent):
    """Process a single query: search -> scrape -> summarize"""
    urls = google_search(query)
    summary, pages_used = summarize_pages(urls, topic, intent)
    return {"query": query, "summary": summary, "pages": pages_used}

def run_refined_queries(queries, topic, intent):
    """Run refined queries in parallel"""
    print(f"🚀 Running {len(queries)} refined queries in parallel...")
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_query, q, topic, intent): q for q in queries}
        for future in as_completed(futures):
            query = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing query '{query}': {e}")
    return results

def generate_course_content(summaries, topic, intent):
    """Generate structured course content with video-search optimized subtopics"""
    print("📚 Creating course with video-search optimized subtopics...")
    combined = "\n\n".join([s['summary'] for s in summaries])
    
    prompt = [
    {
        "role": "system",
        "content": f"""
You are a course designer creating a focused and effective course based ONLY on the given research summary. Your goal is to help the learner fully understand the topic by the end of the course.

Strictly follow these rules:

Context:
Topic: "{topic}"
Perspective/Intent: "{intent}"


Output Format:
# [Course Title]
- [Subtopic 1]
- [Subtopic 2]
...

Output Requirements:
1. Course Title: Create a clear, creative, and **short** title that is highly YouTube-searchable let this be small no need to add extra.
2. Subtopics List: Generate a list of concise, **2–4 word** subtopics.

Subtopic Guidelines:
- Each subtopic must be **YouTube-search friendly** when combined with the topic
- DO NOT include vague or motivational subtopics (e.g., "Stay Focused", "Why It Matters")
- Use only **concepts, keywords, or patterns from the research summary**
- DO NOT use your own external knowledge. Only use what is present in the research summary.
- You may generate as many subtopics as needed to complete the learning intent

DO NOT add any intros, explanations, or section headers other than the title and bullet list.
"""
    },
    {
        "role": "user",
        "content": f"""
Research Summary Content:
{combined}
"""
    }
]

    return azure_chat_completion(prompt)

# Update the research_pipeline function to use the course generator
def research_pipeline(topic, intent):
    """Main research pipeline"""
    print(f"\n{'='*50}")
    print(f"🔬 STARTING RESEARCH")
    print(f"📋 Topic: {topic}")
    print(f"🎯 Intent: {intent}")
    print(f"{'='*50}\n")
    
    # Layer 1: Intent-based initial research
    print(f"\n{'='*30}")
    print(f" LAYER 1: INTENT-FOCUSED RESEARCH ")
    print(f"{'='*30}")
    
    initial_query = generate_intent_based_query(topic, intent)
    urls = google_search(initial_query)
    layer1_summary, layer1_pages = summarize_pages(urls, topic, intent)
    
    if not layer1_summary:
        return "❌ No results found in initial search"
    
    # Layer 2: Refined research based on intent
    print(f"\n{'='*30}")
    print(f" LAYER 2: DEEP DIVE RESEARCH ")
    print(f"{'='*30}")
    refined_queries = refine_queries(layer1_summary, topic, intent)
    refined_results = run_refined_queries(refined_queries, topic, intent)
    
    # Prepare all summaries for course creation
    all_summaries = [
        {"source": "Primary Intent-Based Research", "summary": layer1_summary, "pages": layer1_pages}
    ]
    for res in refined_results:
        all_summaries.append({
            "source": res["query"],
            "summary": res["summary"],
            "pages": res["pages"]
        })
    
    # Generate course content
    print(f"\n{'='*30}")
    print(f" COURSE CREATION ")
    print(f"{'='*30}")
    course_content = generate_course_content(all_summaries, topic, intent)
    
    # Create detailed report
    total_layer2_pages = sum(res['pages'] for res in refined_results)
    total_pages = layer1_pages + total_layer2_pages
    
    report = f"\n{'='*60}\n"
    report += "🎓 INTENT-FOCUSED LEARNING COURSE\n"
    report += f"{'='*60}\n\n"
    report += f"📋 TOPIC: {topic}\n"
    report += f"🎯 PERSPECTIVE: {intent}\n\n"
    report += f"📊 RESEARCH BASIS:\n"
    report += f"- Websites analyzed: {total_pages}\n"
    report += f"- Search queries executed: {1 + len(refined_queries)}\n\n"
    report += f"{'='*60}\n"
    report += "📚 COURSE CONTENT\n"
    report += f"{'='*60}\n\n"
    report += course_content
    
    return report

def print_token_summary():
    """Print final token usage statistics"""
    total = total_prompt_tokens + total_completion_tokens
    print(f"\n{'='*40}")
    print("📦 FINAL TOKEN USAGE SUMMARY")
    print(f"{'='*40}")
    print(f"📏 Total prompt tokens: {total_prompt_tokens:,}")
    print(f"📏 Total completion tokens: {total_completion_tokens:,}")
    print(f"📊 Grand total: {total:,} tokens")
    print(f"{'='*40}")

# === Main Application ===
async def main():
    """Main application flow: Chat -> Extract Intent -> Research"""
    try:
        # Step 1: Extract intent through chat
        extracted_data = await extract_intent_chat()
        
        if not extracted_data:
            print("👋 Research session cancelled.")
            return
        
        topic = extracted_data["topic"]
        intent = extracted_data["intent"]
        
        print(f"\n🚀 Starting deep research...")
        print(f"📋 Topic: {topic}")
        print(f"🎯 Intent: {intent}")
        
        # Step 2: Run research pipeline
        result = research_pipeline(topic, intent)
        
        # Step 3: Display results
        print(result)
        print_token_summary()
        
        # Step 4: Save results to file
        filename = f"research_report_{topic.replace(' ', '_')[:20]}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"\n💾 Research report saved to: {filename}")
        
    except KeyboardInterrupt:
        print("\n👋 Research interrupted by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    asyncio.run(main())