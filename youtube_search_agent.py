from googleapiclient.discovery import build
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY") or "AIzaSyBJo08Dpf3we5cqo9ioNVFVTxuzf-UNaVs"
MAX_RESULTS_PER_SUBTOPIC = 9

json = {
  "topic": "Dribbling Mastery in Football",
  "subtopics": [
    "Close Control Dribbling",
    "Fast Footwork Drills",
    "Agility Enhancement",
    "Step-Over Technique",
    "Body Feint Moves",
    "Scissors Skill Tutorial",
    "Speed Dribbling Drills",
    "Cone Weaving Exercises",
    "Figure Eight Drill",
    "Slalom Dribbling",
    "Ball Control Challenges",
    "Direction Change Drills"
  ]
}




def get_video_durations(video_ids, api_key):
    youtube = build("youtube", "v3", developerKey=api_key)
    request = youtube.videos().list(
        part="contentDetails",
        id=",".join(video_ids)
    )
    response = request.execute()
    durations = {}
    for item in response["items"]:
        video_id = item["id"]
        duration_str = item["contentDetails"]["duration"]
        # Parse ISO 8601 duration (e.g., PT45S, PT1M30S, PT2M)
        seconds = 0
        import re
        m = re.match(r'PT(?:(\d+)M)?(?:(\d+)S)?', duration_str)
        if m:
            minutes = int(m.group(1)) if m.group(1) else 0
            secs = int(m.group(2)) if m.group(2) else 0
            seconds = minutes * 60 + secs
        durations[video_id] = seconds
    return durations

def search_youtube_videos(topic, subtopic, max_results, api_key):
    query = f"{subtopic} {topic}"  # 🔄 Subtopic first
    print(f"\n🔍 Searching YouTube for: '{query}'")

    youtube = build("youtube", "v3", developerKey=api_key)
    search_request = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        maxResults=max_results * 3,
        regionCode="US",               # 🇺🇸 Prefer US-based content
        relevanceLanguage="en"         # 🗣 Prefer English content
    )
    response = search_request.execute()

    video_ids = [item["id"]["videoId"] for item in response["items"]]
    durations = get_video_durations(video_ids, api_key)

    videos = []
    for item in response["items"]:
        video_id = item["id"]["videoId"]
        if durations.get(video_id, 9999) < 60:
            continue  # ⛔ Skip Shorts

        title = item["snippet"]["title"]
        channel = item["snippet"]["channelTitle"]
        published = item["snippet"]["publishedAt"]
        url = f"https://www.youtube.com/watch?v={video_id}"

        # Relevance scoring
        score_topic = fuzz.partial_token_sort_ratio(topic.lower(), title.lower())
        score_sub = fuzz.partial_token_sort_ratio(subtopic.lower(), title.lower())
        match = 1
        if score_topic > 40 and score_sub > 60:
            match = 4
        elif score_topic > 30 and score_sub > 40:
            match = 3
        elif score_sub > 60:
            match = 3
        elif score_topic > 50:
            match = 2

        videos.append({
            "title": title,
            "url": url,
            "channel": channel,
            "published": published,
            "match": match
        })

    videos.sort(key=lambda v: (-v["match"], v["title"]))
    return videos[:max_results]

def run():
    topic = json["topic"]
    subtopics = json["subtopics"]
    all_results = {}

    for subtopic in subtopics:
        print(f"\n📚 Subtopic: {subtopic}")
        videos = search_youtube_videos(topic, subtopic, MAX_RESULTS_PER_SUBTOPIC, API_KEY)
        all_results[subtopic] = videos

        for i, vid in enumerate(videos, 1):
            print(f"{i}. {vid['title']} ({vid['channel']})")
            print(f"   🔗 {vid['url']}")

    return all_results

if __name__ == "__main__":
    run()
