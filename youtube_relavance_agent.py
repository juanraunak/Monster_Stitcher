import os
import json
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from googleapiclient.discovery import build
from openai import AzureOpenAI
from googleapiclient.discovery_cache.base import Cache

# ---------------------- Load Env Vars ----------------------
load_dotenv()

YOUTUBE_API_KEY = "AIzaSyBJo08Dpf3we5cqo9ioNVFVTxuzf-UNaVs"
AZURE_OPENAI_KEY = "2be1544b3dc14327b60a870fe8b94f35"
AZURE_OPENAI_ENDPOINT = "https://notedai.openai.azure.com"
AZURE_OPENAI_VERSION = "2024-06-01"
AZURE_DEPLOYMENT_NAME = "gpt-4o"

if not all([YOUTUBE_API_KEY, AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_VERSION, AZURE_DEPLOYMENT_NAME]):
    raise ValueError("Missing environment variables")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# ---------------------- Concept Map ----------------------
concept_map =  {
  "key_concepts": [
    {
      "name": "Syntax",
      "definition": "The set of rules that defines the combinations of symbols that are considered to be correctly structured programs in the language.",
      "sub_concepts": [
        {
          "name": "Indentation",
          "definition": "Python uses indentation to indicate a block of code."
        },
        {
          "name": "Comments",
          "definition": "Non-executable statements used to explain code."
        }
      ]
    },
    {
      "name": "Data Types",
      "definition": "Classification of data items.",
      "sub_concepts": [
        {
          "name": "Primitive Data Types",
          "definition": "Basic data types such as int, float, str, and bool."
        },
        {
          "name": "Collections",
          "definition": "Data types that store multiple items, such as list, tuple, set, and dict."
        }
      ]
    },
    {
      "name": "Control Flow",
      "definition": "The order in which individual statements, instructions or function calls are executed or evaluated.",
      "sub_concepts": [
        {
          "name": "Conditional Statements",
          "definition": "if, elif, and else statements."
        },
        {
          "name": "Loops",
          "definition": "for and while loops."
        }
      ]
    },
    {
      "name": "Functions",
      "definition": "A block of organized, reusable code that is used to perform a single, related action.",
      "sub_concepts": [
        {
          "name": "Defining Functions",
          "definition": "Using the def keyword."
        },
        {
          "name": "Arguments and Parameters",
          "definition": "Values passed to functions and variables used in function definitions."
        }
      ]
    },
    {
      "name": "Modules and Packages",
      "definition": "Ways to organize and reuse code.",
      "sub_concepts": [
        {
          "name": "Importing Modules",
          "definition": "Using the import statement."
        },
        {
          "name": "Creating Packages",
          "definition": "Organizing modules into directories."
        }
      ]
    },
    {
      "name": "Object-Oriented Programming",
      "definition": "A programming paradigm based on the concept of objects.",
      "sub_concepts": [
        {
          "name": "Classes and Objects",
          "definition": "Blueprints for creating objects and instances of classes."
        },
        {
          "name": "Inheritance",
          "definition": "Mechanism to create a new class using details of an existing class."
        }
      ]
    }
  ],
  "include_keywords": [
    "Python",
    "syntax",
    "data types",
    "control flow",
    "functions",
    "modules",
    "packages",
    "object-oriented programming",
    "classes",
    "inheritance"
  ],
  "exclude_keywords": [
    "CBSE",
    "9th grade",
    "school level"
  ],
  "formulas": [],
  "common_misconceptions": [
    "Python is only for beginners.",
    "Python is too slow for serious development.",
    "Indentation is optional in Python."
  ],
  "notes": [
    "Python is a versatile language used in various fields such as web development, data science, artificial intelligence, and more."
  ]
}
{
  "key_concepts": [
    {
      "name": "Syntax",
      "definition": "The set of rules that defines the combinations of symbols that are considered to be correctly structured programs in the language.",
      "sub_concepts": [
        {
          "name": "Indentation",
          "definition": "Python uses indentation to indicate a block of code."
        },
        {
          "name": "Comments",
          "definition": "Non-executable statements used to explain code."
        }
      ]
    },
    {
      "name": "Data Types",
      "definition": "Classification of data items.",
      "sub_concepts": [
        {
          "name": "Primitive Data Types",
          "definition": "Basic data types such as int, float, str, and bool."
        },
        {
          "name": "Collections",
          "definition": "Data types that store multiple items, such as list, tuple, set, and dict."
        }
      ]
    },
    {
      "name": "Control Flow",
      "definition": "The order in which individual statements, instructions or function calls are executed or evaluated.",
      "sub_concepts": [
        {
          "name": "Conditional Statements",
          "definition": "if, elif, and else statements."
        },
        {
          "name": "Loops",
          "definition": "for and while loops."
        }
      ]
    },
    {
      "name": "Functions",
      "definition": "A block of organized, reusable code that is used to perform a single, related action.",
      "sub_concepts": [
        {
          "name": "Defining Functions",
          "definition": "Using the def keyword."
        },
        {
          "name": "Arguments and Parameters",
          "definition": "Values passed to functions and variables used in function definitions."
        }
      ]
    },
    {
      "name": "Modules and Packages",
      "definition": "Ways to organize and reuse code.",
      "sub_concepts": [
        {
          "name": "Importing Modules",
          "definition": "Using the import statement."
        },
        {
          "name": "Creating Packages",
          "definition": "Organizing modules into directories."
        }
      ]
    },
    {
      "name": "Object-Oriented Programming",
      "definition": "A programming paradigm based on the concept of objects.",
      "sub_concepts": [
        {
          "name": "Classes and Objects",
          "definition": "Blueprints for creating objects and instances of classes."
        },
        {
          "name": "Inheritance",
          "definition": "Mechanism to create a new class using details of an existing class."
        }
      ]
    }
  ],
  "include_keywords": [
    "Python",
    "syntax",
    "data types",
    "control flow",
    "functions",
    "modules",
    "packages",
    "object-oriented programming",
    "classes",
    "inheritance"
  ],
  "exclude_keywords": [
    "CBSE",
    "9th grade",
    "school level"
  ],
  "formulas": [],
  "common_misconceptions": [
    "Python is only for beginners.",
    "Python is too slow for serious development.",
    "Indentation is optional in Python."
  ],
  "notes": [
    "Python is a versatile language used in various fields such as web development, data science, artificial intelligence, and more."
  ]
}

# ---------------------- Disable Cache ----------------------
class NoCache(Cache):
    def get(self, url):
        return None
    def set(self, url, content):
        pass

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY, cache=NoCache())

# ---------------------- YouTube Helpers ----------------------
def search_youtube(query, max_results=30):
    request = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        maxResults=max_results
    )
    response = request.execute()
    return response.get("items", [])

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception as e:
        print(f"Error fetching transcript for {video_id}: {e}")
        return None

# ---------------------- Relevance Evaluation ----------------------
def evaluate_transcript_with_map(transcript, topic, subtopics):
    import json
    subtopic_list = "\n".join([f"- {s}" for s in subtopics])
    prompt = f"""
You are an expert content analyzer. Your task is to evaluate how relevant a video transcript is to a given concept map on a specific topic.

Input:
- Concept Map: a structured JSON with key concepts, sub-concepts, formulas, misconceptions, and notes.
- Transcript Text: the full text of the video transcript.

Steps:
1. Compare the transcript content to the concept map.
2. Identify how many key elements from each category are covered in the transcript.
3. Calculate coverage percentages for each category.
4. Use these weights to compute an overall relevance score (0 to 100):

   - Key Concepts: 40%
   - Sub-Concepts: 30%
   - Formulas: 15%
   - Common Misconceptions: 10%
   - Notes: 5%

5. Provide a clear summary explaining which important concepts or formulas are covered or missing, and why this affects the video’s relevance.

Output:
- Relevance Score: a number from 0 to 10.
- Summary: 2-4 sentences explaining why the video is relevant or not based on how well it covers the concept map.

---

Here is the concept map JSON:

{json.dumps(concept_map)}

Here is the transcript text:

{transcript}

Please analyze and return:

{{
  "relevance_score": "<your rating of 10 for example (7.88/10)>",
  "summary": "<your explanation here>"
}}

"""

    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.3
        )
        # Sometimes GPT may output text before the JSON. Try to find the JSON part:
        content = response.choices[0].message.content.strip()
        # Find first { and last } to extract JSON substring robustly
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            json_str = content[start:end+1]
            return json.loads(json_str)
        else:
            print("⚠️ GPT output did not contain valid JSON.")
            return None

    except Exception as e:
        print(f"Error analyzing transcript: {e}")
        return None

# ---------------------- Main Agent ----------------------
def find_a_level_videos():
    query = "AS Level Physics Forces"
    videos = search_youtube(query)
    results = []

    for video in videos:
        video_id = video.get("id", {}).get("videoId")
        title = video.get("snippet", {}).get("title", "No Title")
        if not video_id:
            continue

        print(f"\n🔍 Checking: {title}")

        transcript = get_transcript(video_id)
        if not transcript:
            print("⚠️ No transcript available. Skipping.")
            continue

        transcript_lower = transcript.lower()
        # Exclude unwanted content
        if any(keyword.lower() in transcript_lower for keyword in concept_map["exclude_keywords"]):
            print("⚠️ Excluded keywords found in transcript. Skipping.")
            continue

        # Include only if relevant keywords exist
        if not any(keyword.lower() in transcript_lower for keyword in concept_map["include_keywords"]):
            print("⚠️ Relevant keywords missing in transcript. Skipping.")
            continue

        try:
            evaluation = evaluate_transcript_with_map(
                transcript=transcript,
                topic="AS Level Physics Forces",
                subtopics=[sc["name"] for kc in concept_map["key_concepts"] for sc in kc.get("sub_concepts", [])]
            )
        except Exception as e:
            print(f"❌ Error analyzing transcript: {e}")
            continue

        if not evaluation:
            print("❌ Invalid analysis result. Skipping.")
            continue

        # Use relevance_score key and convert to float for comparison
        relevance_score_raw = evaluation.get("relevance_score", "0")
        try:
            # Sometimes the score might come like '7.88/10' or '7.9'
            relevance_score = float(str(relevance_score_raw).split("/")[0])
        except:
            relevance_score = 0

        if relevance_score < 5:  # threshold to skip low relevance
            print(f"⚠️ Relevance too low ({relevance_score}/10). Skipping.")
            continue

        print(f"✅ Added with relevance {relevance_score}/10.")

        results.append({
            "video_id": video_id,
            "title": title,
            "summary": evaluation.get("summary", ""),
            "relevance_score": relevance_score
        })

        if len(results) >= 4:
            break

    return results

# ---------------------- Execute ----------------------
if __name__ == "__main__":
    final_videos = find_a_level_videos()
    print("\n✅ Final JSON with up to 4 relevant videos:\n")
    print(json.dumps(final_videos, indent=2))
