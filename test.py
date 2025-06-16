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
import re
from fuzzywuzzy import fuzz
from youtubesearchpython import VideosSearch

# Load environment variables
load_dotenv()

#Query and website count
q=10
websites=5

# YouTube search settings
min_duration = 60  # minimum duration in seconds
max_results_per_topic = 9

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

# === YouTube Search Functions ===
def duration_to_seconds(duration_str):
    """Convert duration string (e.g., '10:30', '1:05:30') to seconds"""
    try:
        if not duration_str or duration_str == "0:00":
            return 0
        
        # Handle different duration formats
        parts = duration_str.split(':')
        if len(parts) == 2:  # MM:SS
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        else:
            return 0
    except (ValueError, AttributeError):
        return None

def extract_topic_shortform(topic):
    """Extract a short form of the topic for YouTube searches"""
    # Split by common separators and take the main keyword
    separators = [' in the ', ' of ', ' from ', ' about ', ' on ', ' - ', ': ']
    
    topic_clean = topic.lower().strip()
    
    # Try to find the main subject before common separators
    for sep in separators:
        if sep in topic_clean:
            topic_clean = topic_clean.split(sep)[0].strip()
            break
    
    # Remove common filler words and keep main keywords
    filler_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
    words = topic_clean.split()
    important_words = [word for word in words if word not in filler_words and len(word) > 2]
    
    # Return the first 1-2 most important words
    if len(important_words) >= 2:
        return ' '.join(important_words[:2])
    elif len(important_words) == 1:
        return important_words[0]
    else:
        # Fallback to first word of original topic
        return topic.split()[0] if topic.split() else topic

def search_youtube(topic_shortform, subtopic_name, max_results):
    """Search YouTube for videos using just topic shortform + subtopic"""
    query = f"{topic_shortform} {subtopic_name}"
    print(f"  🔍 Searching YouTube: '{query}'")
    
    try:
        # Using VideosSearch from youtubesearchpython
        videos_search = VideosSearch(query, limit=max_results * 3)  # Get more results to filter
        results = videos_search.result()['result']
        
        filtered_results = []
        for video in results[:max_results * 3]:
            title = video.get("title", "")
            duration_str = video.get("duration", "0:00")
            duration = duration_to_seconds(duration_str)
            
            # Add basic video info
            video_info = {
                "title": title,
                "url": video.get("link", ""),
                "duration": duration_str,
                "channel": video.get("channel", {}).get("name", ""),
                "views": video.get("viewCount", {}).get("text", ""),
                "published": video.get("publishedTime", ""),
                "match": 1
            }
            
            # Score the video relevance with simpler matching
            if duration is not None and duration > min_duration:
                score_topic = fuzz.partial_token_sort_ratio(topic_shortform.lower(), title.lower())
                score_subtopic = fuzz.partial_token_sort_ratio(subtopic_name.lower(), title.lower())
                
                if score_topic > 40 and score_subtopic > 60:       # highest priority
                    video_info["match"] = 4
                elif score_topic > 30 and score_subtopic > 40:     # middle priority 
                    video_info["match"] = 3
                elif score_subtopic > 60:                          # subtopic focused
                    video_info["match"] = 3
                elif score_topic > 50:                             # topic focused
                    video_info["match"] = 2
                else:                                               # lower priority
                    video_info["match"] = 1
            elif duration is None or duration <= 0:
                video_info["match"] = 1
            
            filtered_results.append(video_info)
        
        # Sort by match score and limit results
        filtered_results.sort(key=lambda x: (-x.get("match", 0), x.get("title", "")))
        final_results = filtered_results[:max_results]
        
        # Add rank
        for index, video in enumerate(final_results):
            video['rank'] = index + 1
        
        print(f"    ✅ Found {len(final_results)} relevant videos for '{query}'")
        return final_results
    
    except Exception as e:
        print(f"    ❌ Error searching YouTube for '{query}': {str(e)}")
        return []

def process_subtopic_youtube(topic_shortform, subtopic_name, max_results):
    """Process a single subtopic for YouTube search"""
    print(f"  📺 Processing subtopic: {subtopic_name}")
    videos = search_youtube(topic_shortform, subtopic_name, max_results)
    return subtopic_name, videos

def search_youtube_for_course(topic, course_content, max_results=5):
    """Search YouTube videos for all subtopics in the generated course"""
    print(f"\n{'='*50}")
    print(f"📺 SEARCHING YOUTUBE VIDEOS FOR COURSE CONTENT")
    print(f"{'='*50}")
    
    # Extract topic shortform for more focused searches
    topic_shortform = extract_topic_shortform(topic)
    print(f"🎯 Using topic shortform: '{topic_shortform}' (from: '{topic}')")
    
    # Parse course content to extract subtopics
    subtopics = []
    lines = course_content.split('\n')
    
    for line in lines:
        line = line.strip()
        # Look for lines that start with "- " (subtopic format)
        if line.startswith('- ') and ':' in line:
            # Extract subtopic name (everything before the first colon)
            subtopic_name = line[2:].split(':')[0].strip()
            if subtopic_name and len(subtopic_name) > 2:
                subtopics.append(subtopic_name)
    
    if not subtopics:
        print("⚠️ No subtopics found in course content for YouTube search")
        return {}
    
    print(f"🎯 Found {len(subtopics)} subtopics to search:")
    for i, subtopic in enumerate(subtopics, 1):
        print(f"  {i}. {subtopic}")
    
    # Search YouTube for each subtopic in parallel
    youtube_results = {}
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        future_to_subtopic = {
            executor.submit(process_subtopic_youtube, topic_shortform, subtopic, max_results): subtopic 
            for subtopic in subtopics
        }
        
        # Collect results
        for future in as_completed(future_to_subtopic):
            subtopic = future_to_subtopic[future]
            try:
                subtopic_name, videos = future.result()
                youtube_results[subtopic_name] = videos
            except Exception as e:
                print(f"❌ Error processing subtopic '{subtopic}': {e}")
                youtube_results[subtopic] = []
    
    return youtube_results

def generate_youtube_report(topic, youtube_results):
    """Generate a simplified report with just subtopics and URLs"""
    if not youtube_results:
        return "\n❌ No YouTube videos found for course subtopics.\n"
    
    report = f"\n{'='*60}\n"
    report += "📺 YOUTUBE VIDEO LINKS\n"
    report += f"{'='*60}\n\n"
    
    total_videos = 0
    for subtopic, videos in youtube_results.items():
        if videos:
            report += f"🎬 {subtopic.upper()}\n"
            for video in videos:
                report += f"{video['url']}\n"
                total_videos += 1
            report += "\n"
        else:
            report += f"🎬 {subtopic.upper()}\n"
            report += "❌ No videos found\n\n"
    
    report += f"📊 Total videos: {total_videos}\n"
    
    return report

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
        "content": """You are a learning intent extractor. Your job is to help users define:

{
  "topic": "",
  "intent": ""
}

- "topic" = what they want to learn about (brief subject)
- "intent" = the perspective or lens they want to study it from (e.g., emotional journey, leadership qualities, communication with God)

Ask only short, helpful questions if needed to clarify either. Once both fields are clear:
1. Output the JSON in this exact format: {"topic": "...", "intent": "..."}
2. Then say: "JSON filled. Starting research..."

Don't ask why they want to learn it. End immediately after printing the JSON."""
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

Based on this summary, generate 10 specific search queries that will find MORE information specifically about the USER'S INTENT. Each query should:
- Target a different aspect of the user's intent (VERY IMPORTANT)
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
You are a course designer creating video-friendly learning content. Structure the course with concise subtopics optimized for YouTube searches.

Topic: "{topic}"
Perspective/Intent: "{intent}"

Output Requirements:
1. Course Title: A very simple name that is youtube friendly 
2. Subtopic List Only: Generate a list of subtopics only names that are youtube friendly

Subtopic Format:
- [Short Subtopic Name]: [1-2 sentence practical explanation]

Guidelines:
- You can generate as many subtopics as needed based on the research summary
- Subtopic names MUST be 2–4 words max
- Subtopic names should be highly YouTube-searchable when combined with the main topic
  (e.g., "Python" + "Lists" → "Python Lists", not "Deep Dive into Python Lists")
- Use ONLY insights found in the research summary
- Do NOT include intros, conclusions, or sections other than the titled list

Output Format:
# [Course Title]
- [Subtopic 1]:
- [Subtopic 2]:
...
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
    
    # Search YouTube for course content
    youtube_results = search_youtube_for_course(topic, course_content, max_results_per_topic)
    youtube_report = generate_youtube_report(topic, youtube_results)
    
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
    report += youtube_report
    
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
    """Main application flow: Chat -> Extract Intent -> Research -> YouTube Search"""
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
        
        # Step 2: Run research pipeline (now includes YouTube search)
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