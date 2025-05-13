from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpStatus

# -----------------------------
# Input Setup (User-defined parameters)
# -----------------------------
topics = ['Topic1', 'Topic2', 'Topic3']  # List of topics (changeable)
topic_durations = {'Topic1': 70, 'Topic2': 30, 'Topic3': 60}  # Duration for each topic (changeable)
daily_time = 60  # User available time per day in minutes (can be adjusted)
num_days = 6      # Number of days user gives to complete the course (changeable)

# -----------------------------
# Chunk Calculation Logic
# -----------------------------
chunks = []  # List to hold all chunks
chunk_to_topic = {}  # Mapping from chunk index to the topic it's assigned to

# We go through each topic and divide it into chunks based on daily_time
chunk_index = 0
topic_part_index = {}  # Dictionary to keep track of part numbers for each topic
for topic, duration in topic_durations.items():
    # Initialize topic part index
    topic_part_index[topic] = 1
    
    # Determine number of chunks for this topic
    num_chunks_for_topic = (duration + daily_time - 1) // daily_time  # Ceiling division
    
    # For each chunk, we'll assign the duration part of the topic
    for i in range(num_chunks_for_topic):
        # The last chunk will handle the remaining time
        chunk_duration = min(daily_time, duration - i * daily_time)
        chunks.append(chunk_duration)
        chunk_to_topic[chunk_index] = topic
        chunk_index += 1

# Now we have the chunks list, where each element is the chunk duration
# and chunk_to_topic mapping from chunk index to topic name.

# -----------------------------
# Create Model
# -----------------------------
model = LpProblem("Topic_Scheduling", LpMinimize)

# -----------------------------
# Variables
# -----------------------------
# x[chunk][d] = 1 if chunk is scheduled on day d
x = {
    (chunk, d): LpVariable(f"x_{chunk}_{d}", cat=LpBinary)
    for chunk in range(len(chunks)) for d in range(1, num_days + 1)
}

# slack[d] = unused time on day d
slack = {d: LpVariable(f"slack_{d}", lowBound=0) for d in range(1, num_days + 1)}

# y[d] = 1 if day d is used
y = {d: LpVariable(f"y_{d}", cat=LpBinary) for d in range(1, num_days + 1)}

# -----------------------------
# Objective: Minimize total slack (wasted time)
# -----------------------------
model += lpSum(slack[d] for d in range(1, num_days + 1)), "Minimize_Total_Slack"

# -----------------------------
# Constraint 1: Each chunk must be scheduled exactly once (some day)
# -----------------------------
for chunk in range(len(chunks)):
    model += lpSum(x[chunk, d] for d in range(1, num_days + 1)) == 1, f"Assign_{chunk}_Once"

# -----------------------------
# Constraint 2: Daily time limit: don't exceed user time
# -----------------------------
for d in range(1, num_days + 1):
    model += (
        lpSum(x[chunk, d] * chunks[chunk] for chunk in range(len(chunks))) + slack[d]
        == daily_time * y[d]
    ), f"DailyTimeLimit_Day{d}"

# -----------------------------
# Constraint 3: Consecutive days only (no gaps)
# -----------------------------
# If day d+1 is used, day d must also be used
for d in range(1, num_days):
    model += y[d + 1] <= y[d], f"No_Gaps_Day{d}"

# -----------------------------
# Constraint 4: Topic order — e.g., Topic1 must be before Topic2
# -----------------------------
# We enforce that the chunks for each topic must be scheduled sequentially
chunk_index = 0
for topic, duration in topic_durations.items():
    num_chunks_for_topic = (duration + daily_time - 1) // daily_time
    topic_chunks = list(range(chunk_index, chunk_index + num_chunks_for_topic))
    chunk_index += num_chunks_for_topic
    
    # Enforce the order of these chunks
    for i in range(len(topic_chunks) - 1):
        model += (
            lpSum(x[topic_chunks[i], d] * d for d in range(1, num_days + 1))
            <= lpSum(x[topic_chunks[i + 1], d] * d for d in range(1, num_days + 1))
        ), f"Prerequisite_{topic_chunks[i]}_before_{topic_chunks[i + 1]}"

# -----------------------------
# Solve the problem
# -----------------------------
model.solve()

# -----------------------------
# Output Results in Readable Format
# -----------------------------
print("Status:", LpStatus[model.status])

# Print the results in readable format
for d in range(1, num_days + 1):
    day_chunks = [chunk for chunk in range(len(chunks)) if x[chunk, d].value() == 1]
    if day_chunks:
        print(f"Day {d}:")
        for chunk in day_chunks:
            topic = chunk_to_topic[chunk]
            part_num = topic_part_index[topic]
            print(f"  {topic} Part {part_num} (Duration: {chunks[chunk]} mins)")
            
            # Increment the part number for the next part of this topic
            topic_part_index[topic] += 1
        print()
