import re
import openai
from datetime import timedelta
from youtube_transcript_api import YouTubeTranscriptApi
import os
from youtube_transcript_api import YouTubeTranscriptApi
import datetime

# --- Azure OpenAI config ---
openai.api_type = "azure"
openai.api_key = "2be1544b3dc14327b60a870fe8b94f35"
openai.azure_endpoint = "https://notedai.openai.azure.com"
openai.api_version = "2024-06-01"
deployment_id = "gpt-4o-mini"  # Your Azure deployment name

# --- Helpers ---


def format_timestamp(seconds):
    """Convert seconds to MM:SS format."""
    return str(datetime.timedelta(seconds=int(seconds)))[2:]

def save_transcript_with_timestamps(video_id, filename='transcript.txt'):
    try:
        # Get transcript data
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)

        lines = []
        for entry in transcript_data:
            timestamp = format_timestamp(entry['start'])
            text = entry['text'].replace('\n', ' ')  # Clean up line breaks
            lines.append(f"{timestamp}\n{text}")

        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"✅ Transcript with timestamps saved to '{filename}'")

    except Exception as e:
        print(f"❌ No transcript available for video ID '{video_id}': {str(e)}")

# Example usage
video_id = "nLRL_NcnK-4"  # Replace with a real YouTube video ID
save_transcript_with_timestamps(video_id)

print("Transcript saved. Checking file content:")
with open("transcript.txt", "r", encoding="utf-8") as f:
    print(f.read()[:1000])  # print first 1000 characters


# --- Daily time input (sample) ---
daily_time_input = {
    "user_id": "user_123",
    "daily_time_minutes": 20
}

def clean_transcript(file_path):
    """
    Reads transcript file, removes filler lines like [Music], and returns cleaned full text.
    """
    filler_pattern = re.compile(r"^\[(music|applause|laughter|silence|noise|background.*|.*inaudible.*)\]$", re.IGNORECASE)
    full_text = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not filler_pattern.match(line):
                full_text.append(line)

    return " ".join(full_text)


def extract_topic_mentions(text, topic):
    """
    Returns all sentences/paragraphs that mention the topic keyword.
    """
    pattern = re.compile(rf"\b{re.escape(topic)}\b", re.IGNORECASE)
    relevant_sections = []

    for paragraph in text.split('. '):
        if pattern.search(paragraph):
            relevant_sections.append(paragraph.strip())

    return ". ".join(relevant_sections)


def parse_time_to_seconds(timestr):
    """
    Converts a time string HH:MM:SS or MM:SS to total seconds.
    """
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


def parse_total_duration(gpt_response):
    """
    Parses the 'Total duration: HH:MM:SS' line from GPT response and returns total seconds.
    """
    import re
    match = re.search(r"Total duration:\s*(\d{1,2}:\d{2}:\d{2})", gpt_response)
    if match:
        timestr = match.group(1)
        return parse_time_to_seconds(timestr)
    else:
        return None


def distill_relevant_segments(text_with_topic_mentions, topic, target_minutes=60):
    """
    Uses GPT to select the most relevant segments about the topic, aiming for roughly target_minutes duration,
    allowing a ±3-4 minute flexibility. If relevant content exceeds the target, mention that explicitly.
    GPT is instructed to calculate total duration explicitly.
    """
    prompt = f"""
You are an assistant that selects the most relevant and informative transcript segments about the topic "{topic}".

Your goal is to select transcript segments that together total about {target_minutes} minutes of spoken content.
It is acceptable for the total duration to vary by ±4 minutes.

For each selected segment, provide:
- A brief summary of what’s being explained
- The time segment in the format [start_time - end_time]

Only include meaningful and insightful discussions. Skip superficial or repeated mentions.

At the end, calculate the total duration of all selected segments and print it in the format:
Total duration: HH:MM:SS

If the total relevant content is significantly longer or shorter than the target, please mention this explicitly.

Transcript:
\"\"\"
{text_with_topic_mentions}
\"\"\"

Output format:

1. Summary: [Brief explanation]
   - Time Segment: [start_time - end_time]

...

Total duration: HH:MM:SS

(If total relevant content is longer or shorter than the target, mention that here.)
"""
    response = openai.chat.completions.create(
        model=deployment_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1200,
    )
    return response.choices[0].message.content


# --- Main ---

def main():
    transcript_file = "transcript.txt"  # Path to your transcript file
    topic = "Loops"  # Change this to your topic of interest
    # Instead of hardcoding 20 minutes
    target_minutes = daily_time_input["daily_time_minutes"]


    print("Reading and cleaning transcript...")
    full_text = clean_transcript(transcript_file)
    print(f"Cleaned transcript length: {len(full_text)} characters")

    print(f"\nExtracting paragraphs mentioning topic: '{topic}'...")
    topic_mentions = extract_topic_mentions(full_text, topic)
    print(f"Filtered topic-related text length: {len(topic_mentions)} characters")

    print(f"\nSending to GPT to get ~{target_minutes} minutes of most relevant content...")
    gpt_summary = distill_relevant_segments(topic_mentions, topic, target_minutes=target_minutes)

    print("\n=== GPT Topic Summary ===\n")
    print(gpt_summary)

    total_seconds = parse_total_duration(gpt_summary)
    if total_seconds is not None:
        print(f"\nParsed total duration from GPT response: {total_seconds // 60} minutes {total_seconds % 60} seconds")
        if total_seconds > (target_minutes + 4) * 60:
            print(f"Note: Total duration exceeds target by more than 4 minutes.")
        elif total_seconds < (target_minutes - 4) * 60:
            print(f"Note: Total duration is less than target by more than 4 minutes.")
        else:
            print(f"Total duration is within acceptable range of the target.")
    else:
        print("\nCould not parse total duration from GPT response.")


if __name__ == "__main__":
    main()
