#  Gym Usage Monitoring System

A machine learning project that predicts daily gym footfall and user activity patterns using **Facebook Prophet**.  
This project helps gym owners analyze trends, optimize resources, and plan better by forecasting future visits.

---

##  Table of Contents
- [About the Project](#about-the-project)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Installation](#installation)

---

##  About the Project

This project focuses on predicting **gym footfall** using historical attendance data.  
By leveraging **Facebook Prophet**, it captures **daily, weekly, and seasonal trends** to provide accurate predictions of upcoming gym visits.

The goal is to enable gyms to:
- Manage staffing and scheduling more effectively  
- Predict low or peak activity days  
- Analyze member engagement trends over time  

---

##  Features

-  Predicts daily gym footfall using historical data  
-  Visualizes trends and seasonal variations  
-  Uses Prophet for time series forecasting  
-  Automatically saves prediction and trend plots as `.png` files  
-  Easy to run and customize for your own dataset  

---

##  Tech Stack

- **Language:** Python  
- **Libraries:** Prophet, Pandas, Matplotlib, Seaborn, NumPy  
- **Environment:** Jupyter Notebook / VS Code  
- **Version Control:** Git & GitHub  

---

##  Dataset

The dataset contains:
- `date` — date of record  
- `duration` — time duration of workout  
- `workout type` — cardio, strength, calisthenics
and more..  

Example:

| User_ID | Date       | Check_In_Time | Check_Out_Time | Duration_Minutes | Day_of_Week | Workout_Type |
|----------|------------|----------------|----------------|------------------|--------------|---------------|
| 1047     | 2025-09-01 | 05:22:00       | 05:54:00       | 32               | Monday       | Cardio        |
| 1006     | 2025-09-01 | 05:33:00       | 06:51:00       | 78               | Monday       | Cardio        |
| 1028     | 2025-09-01 | 07:08:00       | 08:00:00       | 52               | Monday       | Strength      |
| 1035     | 2025-09-01 | 07:33:00       | 08:00:00       | 27               | Monday       | Strength      |
| 1050     | 2025-09-01 | 07:45:00       | 08:00:00       | 15               | Monday       | Strength      |
| 1032     | 2025-09-01 | 07:55:00       | 08:00:00       | 5                | Monday       | Cardio        |
| 1016     | 2025-09-01 | 17:06:00       | 18:05:00       | 59               | Monday       | Cardio        |
| 1037     | 2025-09-01 | 17:37:00       | 18:52:00       | 75               | Monday       | Cardio        |


> You can replace this dataset with your own gym attendance data. Just change the synthetic_gym_data.csv file.

---

##  Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/gym-footfall-predictor.git
   cd gym-footfall-predictor
2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # for Linux/macOS
   venv\Scripts\activate         # for Windows
3. **Install required dependencies**
   ```bash
   pip install -r requirements.txt

Then just run the jupyter notebook (.ipynb file)
