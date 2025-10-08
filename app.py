import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Gym Usage Dashboard")

# --- Title of the App ---
st.title('🏋️‍♂️ Interactive Gym Usage Dashboard')

# --- Data Loading and Caching ---
@st.cache_data
def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"FATAL ERROR: The file '{file_path}' was not found. Please make sure it's in the same directory.")
        return None
    
    # Preprocessing steps
    df['Check_In'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Check_In_Time'].astype(str))
    df['Check_Out'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Check_Out_Time'].astype(str))
    df['WorkoutDuration'] = (df['Check_Out'] - df['Check_In']).dt.total_seconds() / 60
    df['DayOfWeek'] = df['Check_In'].dt.day_name()
    df['HourOfDay'] = df['Check_In'].dt.hour
    
    # Clean Data
    df = df[(df['WorkoutDuration'] >= 10) & (df['WorkoutDuration'] <= 240)].copy()
    return df

# Load the data
FILE_NAME = 'synthetic_gym_data.csv'
df = load_and_preprocess_data(FILE_NAME)

if df is not None:
    # --- NEW: Interactive Sidebar for Filters ---
    st.sidebar.header("Dashboard Filters 🎛️")

    # Filter for Workout Type
    workout_types = ['All'] + sorted(df['Workout_Type'].unique())
    selected_type = st.sidebar.selectbox(
        'Select a Workout Type:',
        workout_types
    )

    # Slider for number of clusters
    num_clusters = st.sidebar.slider(
        'Select Number of User Clusters:',
        min_value=2,
        max_value=5,
        value=3, # Default value
        step=1
    )

    # --- NEW: Filter the data based on selection ---
    if selected_type == 'All':
        filtered_df = df
    else:
        filtered_df = df[df['Workout_Type'] == selected_type]

    # --- Display Sample Data (now filtered) ---
    st.header("Filtered Gym Check-In Data")
    st.write(f"Showing data for: **{selected_type}**")
    st.dataframe(filtered_df[['User_ID', 'WorkoutDuration', 'DayOfWeek', 'HourOfDay', 'Workout_Type']].head())

    # --- Analysis and Visualization Columns ---
    col1, col2 = st.columns(2)

    with col1:
        # --- 1. Analyze Workout Durations (uses filtered_df) ---
        st.header("Distribution of Workout Durations")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.histplot(filtered_df['WorkoutDuration'], bins=30, kde=True, ax=ax1)
        ax1.set_title(f'Duration for {selected_type} Workouts')
        ax1.set_xlabel('Duration (Minutes)')
        ax1.set_ylabel('Number of Visits')
        st.pyplot(fig1)

    with col2:
        # --- 2. User Segments (uses num_clusters) ---
        st.header("User Segments based on Habits")
        user_profiles = df.groupby('User_ID').agg(
            total_visits=('Check_In_Time', 'count'),
            avg_duration=('WorkoutDuration', 'mean')
        ).reset_index()
        
        features = user_profiles[['total_visits', 'avg_duration']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10) # Using the slider value
        user_profiles['cluster'] = kmeans.fit_predict(scaled_features)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=user_profiles,
            x='total_visits',
            y='avg_duration',
            hue='cluster',
            palette='viridis',
            s=100,
            ax=ax2
        )
        ax2.set_title(f'User Segments ({num_clusters} Clusters)')
        ax2.set_xlabel('Total Number of Visits')
        ax2.set_ylabel('Average Workout Duration (Minutes)')
        st.pyplot(fig2)

    # The rest of the app (Heatmap and Prophet) remains the same
    # --- 3. Peak Hours Heatmap (uses filtered_df) ---
    st.header("Gym Usage Heatmap")
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_data = filtered_df.groupby(['HourOfDay', 'DayOfWeek']).size().unstack().reindex(columns=day_order)
    
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap="viridis", linewidths=.5, ax=ax3)
    ax3.set_title(f'Peak Hours for {selected_type} Workouts')
    st.pyplot(fig3)

    # --- 4. Forecasting with Prophet (uses original df) ---
    st.header("Overall Footfall Forecast")
    footfall = df.set_index('Check_In').resample('H').size().reset_index(name='y')
    footfall.rename(columns={'Check_In': 'ds'}, inplace=True)
    
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(footfall)
    
    future_dates = model.make_future_dataframe(periods=3 * 24, freq='H')
    forecast = model.predict(future_dates)
    
    fig4 = model.plot(forecast)
    st.pyplot(fig4)