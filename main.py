import cpmpy as cp
import numpy as np
from datetime import datetime, timedelta

# -----------------------------
# Sample Input
# -----------------------------
course_json = {
    "course": {
        "title": "Sample Course",
        "topics": [
            {"title": "Topic 1", "duration": 10},
            {"title": "Topic 2", "duration": 30},
            {"title": "Topic 3", "duration": 60},
        ]
    }
}

user_json = {
    "user": {
        "username": "john_doe",
        "start_date": "2025-05-01",
        "end_date": "2025-05-02",
        "daily_time": 60
    }
}

# -----------------------------
# Preprocessing
# -----------------------------
topics = course_json["course"]["topics"]
durations = [t["duration"] for t in topics]
num_topics = len(topics)

start_date = datetime.strptime(user_json["user"]["start_date"], "%Y-%m-%d")
end_date = datetime.strptime(user_json["user"]["end_date"], "%Y-%m-%d")
total_days = (end_date - start_date).days + 1
daily_time = user_json["user"]["daily_time"]

# -----------------------------
# Model Setup
# -----------------------------
alloc = cp.intvar(0, max(durations), shape=(total_days, num_topics), name="alloc")
model = cp.Model()

# 1. Each topic must be fully studied
for t in range(num_topics):
    model += (cp.sum(alloc[:, t]) == durations[t])

# 2. Don’t study more than allowed time per day
for d in range(total_days):
    model += (cp.sum(alloc[d, :]) <= daily_time)

# 3. Must finish Topic T before starting Topic T+1
for t in range(num_topics - 1):
    for d in range(total_days):
        model += (cp.sum(alloc[:d+1, t]) < durations[t]).implies(cp.sum(alloc[:d+1, t+1]) == 0)

# -----------------------------
# Solve and Print
# -----------------------------
if model.solve():
    print("🧠 Personalized Learning Plan:")
    for d in range(total_days):
        date_str = (start_date + timedelta(days=d)).strftime("%Y-%m-%d")
        print(f"\n📅 Day {d + 1} ({date_str}):")
        for t in range(num_topics):
            minutes = alloc[d][t].value()
            if minutes > 0:
                print(f"  - {topics[t]['title']} ({minutes} mins)")
else:
    print("❌ No feasible schedule found.")
