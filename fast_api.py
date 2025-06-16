from fastapi import FastAPI, Query
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
import openai
import re
import datetime
from typing import List

app = FastAPI()

# --- Azure OpenAI config ---
openai.api_type = "azure"
openai.api_key = "2be1544b3dc14327b60a870fe8b94f35"
openai.azure_endpoint = "https://notedai.openai.azure.com"
openai.api_version = "2024-06-01"
deployment_id = "gpt-4o-mini"


# --- Helpers ---
def format_timestamp(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))[2:]


def parse_time_to_seconds(timestr):
    parts = timestr.strip().split(':')
    parts = [int(p) for p in parts]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = 0
        m, s = parts
    else:
        return 0
    return h * 3600 + m * 60 + s

def extract_time_segments(gpt_response: str):
    segment_pattern = r"\[(\d{1,2}:\d{2}) - (\d{1,2}:\d{2})\]"
    matches = re.findall(segment_pattern, gpt_response)

    segments = []
    total_seconds = 0
    for start, end in matches:
        segments.append(f"{start} - {end}")
        start_sec = parse_time_to_seconds(start)
        end_sec = parse_time_to_seconds(end)
        duration = max(0, end_sec - start_sec)
        total_seconds += duration

    # Format to "X minutes Y seconds"
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    human_duration = f"{minutes} minutes {seconds} seconds"

    return segments, total_seconds, human_duration

def parse_total_duration(gpt_response):
    match = re.search(r"Total duration:\s*(\d{1,2}:\d{2}:\d{2})", gpt_response)
    if match:
        timestr = match.group(1)
        return parse_time_to_seconds(timestr)
    return None


def clean_transcript_lines(transcript_data):
    filler_pattern = re.compile(r"^\[(music|applause|laughter|silence|noise|background.*|.*inaudible.*)\]$", re.IGNORECASE)
    cleaned = []
    for entry in transcript_data:
        text = entry['text'].replace('\n', ' ').strip()
        if not filler_pattern.match(text.lower()):
            cleaned.append(f"{format_timestamp(entry['start'])}\n{text}")
    return "\n".join(cleaned)


def extract_topic_mentions(text, topic):
    pattern = re.compile(rf"\b{re.escape(topic)}\b", re.IGNORECASE)
    relevant = []
    for paragraph in text.split('. '):
        if pattern.search(paragraph):
            relevant.append(paragraph.strip())
    return ". ".join(relevant)


def distill_relevant_segments(text_with_topic_mentions, topic, target_minutes):
    
    prompt = f"""
    You are an assistant that selects the most relevant and insightful transcript segments related to the topic: "{topic}".

    Your goal is to extract transcript segments that, together, total approximately {target_minutes} minutes of spoken content. A variation of ±4 minutes is acceptable.

    Guidelines:
    - Only select meaningful, educational, or insightful moments where the speaker clearly explains or elaborates on the topic.
    - Avoid superficial mentions, repeated ideas, or segments with little informational value.
    - Try to spread the selections across different parts of the video to cover a range of explanations.

    Output format:
    1. A list of time segments in the format: [MM:SS - MM:SS]
    2. A short explanation (1–2 lines) describing why these were selected.

    Only return:
    - A clean list of all time segments.
    - The explanation under the heading "Why these were selected:".
    - At the end, write "Total duration: HH:MM:SS".

    Transcript:
    \"\"\"{text_with_topic_mentions}\"\"\"
    """

    response = openai.chat.completions.create(
        model=deployment_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1200,
    )
    return response.choices[0].message.content


# --- Request model ---
class TranscriptRequest(BaseModel):
    video_id: str
    topic: str
    daily_time_minutes: int = 20


# --- Endpoint ---
@app.post("/extract-segments/")
def extract_segments(requests: List[TranscriptRequest]):
    results = []

    for request in requests:
        try:
            transcript_data = YouTubeTranscriptApi.get_transcript(request.video_id)
            raw_text = clean_transcript_lines(transcript_data)
            topic_mentions = extract_topic_mentions(raw_text, request.topic)
            gpt_summary = distill_relevant_segments(
                topic_mentions, request.topic, request.daily_time_minutes
            )
            segments, total_seconds, human_duration = extract_time_segments(gpt_summary)

            results.append({
                "video_id": request.video_id,
                "topic": request.topic,
                "target_minutes": request.daily_time_minutes,
                "time_segments": segments,
                "total_duration": human_duration,
                "total_duration_seconds": total_seconds
            })

        except Exception as e:
            results.append({
                "video_id": request.video_id,
                "topic": request.topic,
                "error": str(e)
            })

    return results