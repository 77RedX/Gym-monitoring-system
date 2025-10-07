import pandas as pd
import numpy as np
from datetime import timedelta
import io # Used for demonstrating the CSV output here

N_ENTRIES = 500
START_DATE = pd.to_datetime('2025-10-01')
N_MEMBERS = 50 


# Generate User IDs (Anonymized)
df = pd.DataFrame({
    'User_ID': np.random.randint(1001, 1001 + N_MEMBERS, N_ENTRIES)
})

# Create a probability distribution for hours, biased towards peak gym times (5 PM to 8 PM)
peak_hours = list(range(17, 21)) 
mid_hours = list(range(6, 11)) + list(range(15, 17)) + list(range(21, 23)) 
off_hours = list(range(0, 6)) + list(range(11, 15)) + list(range(23, 24))

# Weight the hours: Peak (3x), Mid (2x), Off (1x)
hours = peak_hours * 3 + mid_hours * 2 + off_hours * 1

# Generate random days, sampled hours, and minutes
days = np.random.randint(0, 21, N_ENTRIES) # 21 days of data
sampled_hours = np.random.choice(hours, N_ENTRIES)
minutes = np.random.randint(0, 60, N_ENTRIES)

# Combine into Check_In timestamps
check_in_datetimes = [
    START_DATE + timedelta(days=int(d), hours=int(h), minutes=int(m)) 
    for d, h, m in zip(days, sampled_hours, minutes)
]
df['Check_In'] = check_in_datetimes

# Simulate Duration
durations_min = np.concatenate([
    np.random.randint(30, 61, int(N_ENTRIES * 0.3)),  # Short (30%)
    np.random.randint(60, 91, int(N_ENTRIES * 0.5)),  # Medium (50%) - Peak Duration
    np.random.randint(90, 121, int(N_ENTRIES * 0.2))  # Long (20%)
])

if len(durations_min) != N_ENTRIES:
    durations_min = durations_min[:N_ENTRIES] 
    
np.random.shuffle(durations_min) 

df['Duration_Minutes'] = durations_min
df['Check_Out'] = df.apply(lambda row: row['Check_In'] + timedelta(minutes=row['Duration_Minutes']), axis=1)

df['Day_of_Week'] = df['Check_In'].dt.day_name()
workout_types = np.random.choice(
    ['Strength', 'Cardio', 'Class', 'Yoga'], 
    N_ENTRIES, 
    p=[0.5, 0.3, 0.15, 0.05] # 50% Strength (Simulated Emerging Trend)
)
df['Workout_Type'] = workout_types

# Sort data and select final columns
final_df = df[['User_ID', 'Check_In', 'Check_Out', 'Duration_Minutes', 'Day_of_Week', 'Workout_Type']]
final_df = final_df.sort_values(by='Check_In').reset_index(drop=True)
#print csv
print("--- CSV Data Start ---")
print(final_df.head().to_csv(index=False))
print("--- CSV Data End (Total entries: 500) ---")
#save csv
final_df.to_csv('synthetic_gym_data.csv', index=False)