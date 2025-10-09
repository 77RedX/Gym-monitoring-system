import pandas as pd
import numpy as np
from datetime import timedelta, datetime

N_ENTRIES = 500
N_MEMBERS = 50
START_DATE = datetime(2025, 9, 1)
FILE_NAME = 'synthetic_gym_data.csv'

def generate_gym_data(n_entries, n_members, start_date):
    df = pd.DataFrame({
        'User_ID': np.random.randint(1001, 1001 + n_members, n_entries)
    })

    peak_hours = list(range(17, 20))
    mid_hours = list(range(5, 8))

    hours = peak_hours * 3 + mid_hours * 2
    
    days = np.random.randint(0, 21, n_entries)
    sampled_hours = np.random.choice(hours, n_entries)
    minutes = np.random.randint(0, 60, n_entries)

    check_in_datetimes = [
        start_date + timedelta(days=int(d), hours=int(h), minutes=int(m)) 
        for d, h, m in zip(days, sampled_hours, minutes)
    ]
    df['Check_In'] = check_in_datetimes
    #print(df['Check_In'])

    p1 = 0.3
    p2 = 0.5
    p3 = 0.2
    durations_min_initial = np.concatenate([
        np.random.randint(30, 61, int(n_entries * p1)),
        np.random.randint(60, 91, int(n_entries * p2)),
        np.random.randint(90, 121, int(n_entries * p3))
    ])

    if len(durations_min_initial) != n_entries:
        durations_min_initial = durations_min_initial[:n_entries]

    np.random.shuffle(durations_min_initial)
    df['Initial_Duration_Minutes'] = durations_min_initial

    def calculate_capped_duration(row):
        check_in = row['Check_In']
        initial_duration_minutes = row['Initial_Duration_Minutes']

        if check_in.hour < 8:
            cap_time = check_in.replace(hour=8, minute=0, second=0, microsecond=0)
        elif check_in.hour < 20:
            cap_time = check_in.replace(hour=20, minute=0, second=0, microsecond=0)
        else:
            return 1

        max_duration_td = cap_time - check_in
        max_duration_minutes = max_duration_td.total_seconds() / 60

        final_duration = min(initial_duration_minutes, max_duration_minutes)

        return max(1, int(final_duration))

    df['Duration_Minutes'] = df.apply(calculate_capped_duration, axis=1)

    df['Check_Out'] = df.apply(lambda row: row['Check_In'] + timedelta(minutes=int(row['Duration_Minutes'])), axis=1)

    df.drop(columns=['Initial_Duration_Minutes'], inplace=True)
    
    df['Date'] = df['Check_In'].dt.date
    df['Check_In_Time'] = df['Check_In'].dt.time
    df['Check_Out_Time'] = df['Check_Out'].dt.time
    
    df['Day_of_Week'] = df['Check_In'].dt.day_name()
    workout_types = np.random.choice(
        ['Strength', 'Cardio', 'Calisthenics', 'Yoga'], 
        n_entries, 
        p=[0.5, 0.3, 0.15, 0.05]
    )
    df['Workout_Type'] = workout_types

    final_cols = [
        'User_ID', 'Date', 'Check_In_Time', 'Check_Out_Time', 
        'Duration_Minutes', 'Day_of_Week', 'Workout_Type'
    ]
    final_df = df[final_cols]
    
    return final_df.sort_values(by=['Date', 'Check_In_Time']).reset_index(drop=True)

if __name__ == "__main__":
    df_generated = generate_gym_data(N_ENTRIES, N_MEMBERS, START_DATE)
    df_generated.to_csv(FILE_NAME, index=False)
    print(f"Successfully generated {len(df_generated)} records with separated Date and Time columns.")
    print(f"File saved as '{FILE_NAME}' in the current directory.")
    print("\nFirst 5 entries:")
    print(df_generated.head())