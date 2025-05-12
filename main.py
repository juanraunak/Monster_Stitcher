import json
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from database.connection import SessionLocal
from database.models import UserPlan

# ------------------- Sample Input -------------------
user_input_json = {
    "user": {
        "username": "john_doe",
        "start_date": "2025-05-01",
        "daily_time": 45
    }
}

course_json = {
    "course_id": "C101",
    "topics": [
        {"topic_id": "T1", "name": "Intro to AI", "duration": 30},
        {"topic_id": "T2", "name": "History of AI", "duration": 20},
        {"topic_id": "T3", "name": "Machine Learning Basics", "duration": 45},
        {"topic_id": "T4", "name": "Neural Networks", "duration": 60}
    ],
    "prerequisite_order": ["T1", "T2", "T3", "T4"]
}
# -----------------------------------------------------

def build_optimized_schedule(user_input, course):
    start_date = datetime.strptime(user_input["user"]["start_date"], "%Y-%m-%d")
    daily_limit = user_input["user"]["daily_time"]
    topics_dict = {t["topic_id"]: t["duration"] for t in course["topics"]}

    schedule = {}
    current_date = start_date
    remaining_time = daily_limit
    day_plan = {"date": current_date.strftime("%Y-%m-%d"), "topics": [], "total_duration": 0, "topic_details": []}

    # Go through each topic in order
    for topic_id in course["prerequisite_order"]:
        topic_duration = topics_dict[topic_id]
        time_left = topic_duration

        while time_left > 0:
            if remaining_time == 0:
                # Save current day and start new day
                schedule[day_plan["date"]] = day_plan
                current_date += timedelta(days=1)
                remaining_time = daily_limit
                day_plan = {"date": current_date.strftime("%Y-%m-%d"), "topics": [], "total_duration": 0, "topic_details": []}

            used_time = min(time_left, remaining_time)
            time_left -= used_time
            remaining_time -= used_time

            # Log topic fragment
            if topic_id not in day_plan["topics"]:
                day_plan["topics"].append(topic_id)

            day_plan["topic_details"].append({
                "topic": topic_id,
                "time_spent": used_time
            })
            day_plan["total_duration"] += used_time

    # Add last day
    if day_plan["total_duration"] > 0:
        schedule[day_plan["date"]] = day_plan

    # Return as sorted list
        sorted_dates = sorted(schedule.keys())

        # Build final output
        final_output = {
            "total_days_spent": len(sorted_dates),
            "schedule": [schedule[date] for date in sorted_dates]
        }

        return final_output
    

# Run and print result
daily_schedule = build_optimized_schedule(user_input_json, course_json)
print(json.dumps(daily_schedule, indent=2))

def save_plan_to_db(plan_json, user_id="user_123"):
    db: Session = SessionLocal()
    try:
        print("🔄 Saving plan to database...")  # This message is shown when the function starts
        new_plan = UserPlan(
            user_id=user_id,
            plan_json=plan_json  # This is where the daily_schedule is passed
        )
        print("Adding to database:", new_plan)  # This will print the new_plan object
        db.add(new_plan)
        db.commit()
        print("✅ Plan saved to database.")
    except Exception as e:
        db.rollback()
        print("❌ Failed to save:", e)
    finally:
        db.close()

# Save the daily schedule (plan) to the database
save_plan_to_db(plan_json=daily_schedule)
