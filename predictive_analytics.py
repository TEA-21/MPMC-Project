import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import datetime
import random

# ============================================================
# FEATURE 5: Predictive Analytics & Smart Occupancy Forecast
# ============================================================

CSV_FILE = 'parking_log.csv'

def generate_synthetic_data():
    """
    Generates realistic 30-day historical seed data if the 
    live server hasn't saved enough real logs yet.
    Simulates morning (8 AM) and evening (6 PM) rush hours.
    """
    print(f"[*] No existing '{CSV_FILE}' found. Generating 30 days of synthetic data for demonstration...")
    base_time = datetime.datetime.now() - datetime.timedelta(days=30)
    
    with open(CSV_FILE, 'w') as f:
        for day in range(30):
            # Morning rush (8:00 - 10:00)
            morning_peak = random.randint(15, 30)
            for _ in range(morning_peak):
                hour = random.randint(8, 10)
                minute = random.randint(0, 59)
                t = base_time + datetime.timedelta(days=day, hours=hour, minutes=minute)
                f.write(f"{t},1\n")
                
            # Evening rush (17:00 - 19:00)
            evening_peak = random.randint(25, 45)
            for _ in range(evening_peak):
                hour = random.randint(17, 19)
                minute = random.randint(0, 59)
                t = base_time + datetime.timedelta(days=day, hours=hour, minutes=minute)
                f.write(f"{t},1\n")
                
            # Random off-peak background traffic
            off_peak = random.randint(5, 15)
            for _ in range(off_peak):
                hour = random.randint(0, 23)
                minute = random.randint(0, 59)
                t = base_time + datetime.timedelta(days=day, hours=hour, minutes=minute)
                f.write(f"{t},1\n")

def main():
    # 1. Setup Data Source
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        generate_synthetic_data()

    print("[*] Loading historical server logs...")
    # The server logs don't have a header, so we explicitly define column names
    df = pd.read_csv(CSV_FILE, names=['timestamp', 'entry'])
    
    # 2. Extract DateTime Features
    # Convert string to fast datetime objects and extract the specific Hour of day
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    
    # 3. Aggregate into Hourly Bins
    # Find total entries per hour across each specific day independently
    daily_hourly_counts = df.groupby(['date', 'hour'])['entry'].sum().reset_index()
    
    # Average the entries across all days to build a stable 24-hour profile
    avg_hourly = daily_hourly_counts.groupby('hour')['entry'].mean().reset_index()
    
    # Pad missing hours with 0 to ensure the model sees a complete 0-23 timeline
    all_hours = pd.DataFrame({'hour': np.arange(24)})
    avg_hourly = pd.merge(all_hours, avg_hourly, on='hour', how='left').fillna(0)
    
    X_train = avg_hourly[['hour']].values
    y_train = avg_hourly['entry'].values

    # 4. Scikit-Learn Modeling
    print("[*] Training predictive analytics model...")
    # A flat Linear Regression would draw a straight line, completely ignoring real 
    # world human traffic patterns (peaks and valleys). 
    # Using a Degree 4 Polynomial transformation captures the natural dual-peak curve 
    # (morning rush & evening outflux) perfectly.
    poly = PolynomialFeatures(degree=4)
    X_poly = poly.fit_transform(X_train)
    
    model = LinearRegression()
    model.fit(X_poly, y_train)
    
    # 5. Extrapolate Smooth Curve
    # Generate 100 continuous points between 0 and 23 to draw a perfectly smooth prediction curve
    X_smooth = np.linspace(0, 23, 100).reshape(-1, 1)
    X_smooth_poly = poly.transform(X_smooth)
    y_smooth = model.predict(X_smooth_poly)
    
    # Sanitize predictions (can't have negative cars)
    y_smooth = np.maximum(y_smooth, 0)
    
    # Identify exact maximum capacity peak mathematically
    peak_idx = np.argmax(y_smooth)
    predicted_peak_time = X_smooth[peak_idx][0]
    
    # Output business analytics to console
    max_real_hour = int(avg_hourly.loc[avg_hourly['entry'].idxmax()]['hour'])
    print(f"\n[INSIGHT] Highest Historical Average: {max_real_hour:02d}:00 hours")
    print(f"[INSIGHT] AI Predicted Mathematical Peak: At approx {predicted_peak_time:.1f} hours")

    # 6. Visualise using Matplotlib
    plt.style.use('dark_background') # Give it a sleek analytics dashboard look
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Raw historical data as semi-transparent bars
    ax.bar(avg_hourly['hour'], avg_hourly['entry'], 
           color='#4facf7', alpha=0.5, label='Historical Avg (Per Hour)')
    
    # Smooth AI Prediction trendline
    ax.plot(X_smooth, y_smooth, 
            color='#ff5757', linewidth=3, label='ML Polynomial Forecast')
    
    # Drop a vertical line precisely on the predicted rush hour peak
    ax.axvline(x=predicted_peak_time, color='#ffcc00', linestyle='--', linewidth=2,
               label=f'Predicted Maximum Capacity ({predicted_peak_time:.1f}H)')
    
    # Graph Formatting
    ax.set_title("Smart Occupancy: 24-Hour Traffic Predictive Analysis", fontsize=16, pad=15)
    ax.set_xlabel("Hour of Day (0-23)", fontsize=12)
    ax.set_ylabel("Average Vehicle Throughput", fontsize=12)
    ax.set_xticks(np.arange(0, 24, 1))
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    
    # Remove top and right spines for a clean HUD look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='upper right', frameon=True, facecolor='#202020', edgecolor='none')
    plt.tight_layout()
    
    print("\n[*] Launching Matplotlib Dashboard window...")
    plt.show()

if __name__ == "__main__":
    main()
