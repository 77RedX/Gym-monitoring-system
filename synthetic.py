import pandas as pd
import numpy as np
from datetime import timedelta, datetime

# --- Configuration ---
N_ENTRIES = 500  # Total number of gym visits to simulate
N_MEMBERS = 50   # Number of unique, anonymized users
START_DATE = datetime(2025, 9, 1) # Start date for the 3-week simulation
FILE_NAME = 'synthetic_gym_data.csv'

def generate_gym_data(n_entries, n_members, start_date):
    """
    Generates a synthetic dataset for gym usage with biased check-in times and durations.
    The output includes separate columns for Date, Check-In Time, and Check-Out Time.

    Args:
        n_entries (int): The number of total visit records.
        n_members (int): The number of unique anonymized users.
        start_date (datetime): The starting date for the data.

    Returns:
        pd.DataFrame: The generated DataFrame.
    """
    # 1. Generate User IDs (Anonymized)
    df = pd.DataFrame({
        'User_ID': np.random.randint(1001, 1001 + n_members, n_entries)
    })

    # --- 2. Generate Check-In/Out Timestamps (Biased towards Peak Hours) ---
    
    # Define time segments and their likelihoods (weights)
    peak_hours = list(range(17, 19))  # 5 PM to 9 PM (highest weight)
    mid_hours = list(range(6, 7)) # Morning, pre-peak, late (medium weight)
    #off_hours = list(range(0, 6)) + list(range(11, 15)) + list(list(range(23, 24))) # Overnight, mid-day lull (lowest weight)

    # Weight the hours: Peak (3x), Mid (2x), Off (1x)
    hours = peak_hours * 5 + mid_hours * 2
    
    # Generate random days (0 to 20), sampled hours, and minutes
    days = np.random.randint(0, 21, n_entries)
    sampled_hours = np.random.choice(hours, n_entries)
    minutes = np.random.randint(0, 60, n_entries)

    # Combine into Check_In timestamps (full datetime objects)
    check_in_datetimes = [
        start_date + timedelta(days=int(d), hours=int(h), minutes=int(m)) 
        for d, h, m in zip(days, sampled_hours, minutes)
    ]
    df['Check_In'] = check_in_datetimes

    # --- 3. Generate Workout Duration and Check-Out Time ---

    # Simulate Duration: most common duration is 60-90 minutes (50%)
    durations_min = np.concatenate([
        np.random.randint(30, 61, int(n_entries * 0.3)),  # Short (30%)
        np.random.randint(60, 91, int(n_entries * 0.5)),  # Medium (50%) - Peak Duration
        np.random.randint(90, 121, int(n_entries * 0.2))  # Long (20%)
    ])

    # Adjust size if necessary
    if len(durations_min) != n_entries:
        durations_min = durations_min[:n_entries] 
        
    np.random.shuffle(durations_min) 
    df['Duration_Minutes'] = durations_min
    df['Check_Out'] = df.apply(lambda row: row['Check_In'] + timedelta(minutes=int(row['Duration_Minutes'])), axis=1)
    # --- 4. Restructure Columns as Requested (Date and Time Separation) ---
    df['Date'] = df['Check_In'].dt.date
    df['Check_In_Time'] = df['Check_In'].dt.time
    df['Check_Out_Time'] = df['Check_Out'].dt.time
    
    # --- 5. Add Optional Features ---
    df['Day_of_Week'] = df['Check_In'].dt.day_name()
    workout_types = np.random.choice(
        ['Strength', 'Cardio', 'Class', 'Yoga'], 
        n_entries, 
        p=[0.5, 0.3, 0.15, 0.05]
    )
    df['Workout_Type'] = workout_types

    # Finalize columns order
    final_cols = [
        'User_ID', 'Date', 'Check_In_Time', 'Check_Out_Time', 
        'Duration_Minutes', 'Day_of_Week', 'Workout_Type'
    ]
    final_df = df[final_cols]
    
    return final_df.sort_values(by=['Date', 'Check_In_Time']).reset_index(drop=True)

if __name__ == "__main__":
    df_generated = generate_gym_data(N_ENTRIES, N_MEMBERS, START_DATE)
    
    # Save the file to the current directory
    df_generated.to_csv(FILE_NAME, index=False)
    
    print(f"Successfully generated {len(df_generated)} records with separated Date and Time columns.")
    print(f"File saved as '{FILE_NAME}' in the current directory.")
    print("\nFirst 5 entries:")
    print(df_generated.head())
