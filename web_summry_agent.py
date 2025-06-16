import requests
import time
from urllib.parse import urlencode
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken

# === Global token counters ===
total_prompt_tokens = 0
total_completion_tokens = 0


# Configuration
GOOGLE_API_KEY = "AIzaSyCk4DYKCm5sSLz63aFUlVk8E04QPSvjXT8"
GOOGLE_CX = "53459b243c2c34e0c"
AZURE_OPENAI_API_KEY = "2be1544b3dc14327ab60a870fe8b94f35"
AZURE_OPENAI_ENDPOINT = "https://notedai.openai.azure.com"
AZURE_OPENAI_API_VERSION = "2024-06-01"
DEPLOYMENT_NAME = "gpt-4o"

headers = {
    "Content-Type": "application/json",
    "api-key": AZURE_OPENAI_API_KEY
}

model = SentenceTransformer("all-MiniLM-L6-v2")
ENCODING = tiktoken.encoding_for_model("gpt-4o")

# Token count
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


# Azure Chat Completion
def azure_chat_completion(messages):
    global total_prompt_tokens, total_completion_tokens

    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"

    # Count prompt tokens
    prompt_tokens = count_tokens_from_messages(messages, model="gpt-4o")

    response = requests.post(url, headers=headers, json={
        "messages": messages,
        "temperature": 0.7
    })
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    completion_tokens = len(ENCODING.encode(content))

    # Update global totals
    total_prompt_tokens += prompt_tokens
    total_completion_tokens += completion_tokens

    print(f"📏 Prompt tokens: {prompt_tokens} | Completion tokens: {completion_tokens} | Total this call: {prompt_tokens + completion_tokens}")
    return content

# Google Search
def google_search(query, num_results=10):
    print(f"🔍 Searching Google: '{query}'")
    query_params = urlencode({"q": query, "cx": GOOGLE_CX, "key": GOOGLE_API_KEY, "num": num_results})
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

# Scrape Page Text
def scrape_page_text(url):
    try:
        print(f"🌐 Scraping: {url[:60]}...")
        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        print(f"❌ Failed to scrape {url}: {str(e)}")
        return ""

# Summarize Pages
def summarize_pages(urls):
    print(f"📚 Summarizing {len(urls)} pages...")
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
        {"role": "system", "content": "You are a research assistant that summarizes information concisely."},
        {"role": "user", "content": f"Summarize the following content into a cohesive overview:\n\n{combined_text}"}
    ]
    summary = azure_chat_completion(prompt)
    print(f"✅ Summary completed ({len(summary)} chars)")
    return summary, len(contents)

# Refine Queries Based on Summary
def refine_queries(summary, topic):
    print("🧠 Generating refined queries...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant skilled at generating deeper research questions."},
        {"role": "user", "content": f"Based on this summary about {topic}, generate 7 deeper or more specific queries that explore subtopics, clarify math, or real-world uses — but stay on topic:\n\n{summary}"}
    ]
    raw_queries = azure_chat_completion(messages)
    queries = [q.strip("-• ") for q in raw_queries.split("\n") if q.strip()]
    print("🔍 Refined queries:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    return queries

# Run Refined Queries in Parallel
def run_refined_queries(queries):
    print(f"🚀 Running {len(queries)} refined queries in parallel...")
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_query, q): q for q in queries}
        for future in as_completed(futures):
            query = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing query '{query}': {e}")
    return results

def process_query(query):
    urls = google_search(query)
    summary, pages_used = summarize_pages(urls)
    return {"query": query, "summary": summary, "pages": pages_used}

# Generate Final Summary
def generate_final_summary(summaries, topic):
    print("🧩 Combining all summaries into final output...")
    combined = "\n\n".join([
        f"## Summary Source {i+1} ##\n{s['summary']}" 
        for i, s in enumerate(summaries)
    ])
    
    prompt = [
{
  "role": "system",
  "content": "You are a senior research assistant synthesizing information from multiple sources."
},
{
  "role": "user",
  "content": f"""
What you have been provided is list of summries form multiple sources this is orginal topic {topic} by combining these research summaries create on final summry:

{combined}

Write a well-organized synthesis that dynamically adapts to the nature of the topic — whether it's narrative, technical, historical, reflective, or practical.

Your summary should:
- Emphasize the most relevant themes, events, concepts, or findings depending on the topic
- Use structure only if it enhances clarity (e.g., sections for concepts, timelines, phases, or themes)
- Reflect depth, clarity, and flow suitable to the topic type
- Highlight key insights, takeaways, or implications where appropriate

Avoid forcing rigid sectioning. Let the content shape the format — narrative flow for stories, structured breakdown for technical material, thematic for reflective topics, or practical strategy for exam/skill prep.
"""
}


    ]
    return azure_chat_completion(prompt)

# Full Pipeline
def research_pipeline(initial_query):
    print(f"\n{'='*50}")
    print(f"🔬 STARTING RESEARCH: {initial_query}")
    print(f"{'='*50}\n")
    
    # Layer 1: Initial research
    print(f"\n{'='*30}")
    print(f" LAYER 1: PRIMARY RESEARCH ")
    print(f"{'='*30}")
    urls = google_search(initial_query)
    layer1_summary, layer1_pages = summarize_pages(urls)
    
    if not layer1_summary:
        return "❌ No results found in initial search"
    
    # Layer 2: Refined research
    print(f"\n{'='*30}")
    print(f" LAYER 2: DEEP DIVE RESEARCH ")
    print(f"{'='*30}")
    refined_queries = refine_queries(layer1_summary, initial_query)
    refined_results = run_refined_queries(refined_queries)
    
    # Prepare all summaries for final combination
    all_summaries = [
        {"source": "Primary Research", "summary": layer1_summary, "pages": layer1_pages}
    ]
    for res in refined_results:
        all_summaries.append({
            "source": res["query"],
            "summary": res["summary"],
            "pages": res["pages"]
        })
    
    # Generate final comprehensive summary
    print(f"\n{'='*30}")
    print(f" FINAL SYNTHESIS ")
    print(f"{'='*30}")
    final_summary = generate_final_summary(all_summaries, initial_query)
    
    # Prepare detailed report
    report = f"\n{'='*50}\n"
    report += "🔎 RESEARCH REPORT\n"
    report += f"{'='*50}\n\n"
    report += f"TOPIC: {initial_query}\n\n"
    
    report += f"📊 LAYER 1 RESULTS:\n"
    report += f"- Websites crawled: {layer1_pages}\n"
    report += f"- Summary: {layer1_summary[:300]}...\n\n"
    
    report += f"🔍 REFINED QUERIES ({len(refined_queries)}):\n"
    for i, q in enumerate(refined_queries, 1):
        report += f"  {i}. {q}\n"
    
    report += f"\n📚 LAYER 2 RESULTS:\n"
    total_layer2_pages = 0
    for i, res in enumerate(refined_results, 1):
        report += f"- Query {i}: '{res['query']}'\n"
        report += f"  Pages: {res['pages']}\n"
        report += f"  Summary: {res['summary'][:200]}...\n"
        total_layer2_pages += res['pages']
    
    report += f"\n📈 TOTAL PAGES ANALYZED: {layer1_pages + total_layer2_pages}\n\n"
    report += f"{'='*50}\n"
    report += "🧠 COMPREHENSIVE FINAL SUMMARY\n"
    report += f"{'='*50}\n\n"
    report += final_summary
    
    return report
def print_token_summary():
    total = total_prompt_tokens + total_completion_tokens
    print("\n📦 FINAL TOKEN USAGE")
    print(f"📏 Total prompt tokens: {total_prompt_tokens}")
    print(f"📏 Total completion tokens: {total_completion_tokens}")
    print(f"📊 Grand total: {total} tokens")

# Example run
if __name__ == "__main__":
    topic = "Spritual isolation according to the Bible what is it"
    result = research_pipeline(topic)
    print(result)
    print_token_summary()
