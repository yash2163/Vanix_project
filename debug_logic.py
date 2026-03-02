import pandas as pd
import numpy as np
import glob
import os

class CattleIntelligenceSystem:
    def __init__(self):
        self.RES_THRESHOLD = 0.35
        self.NIGHT_HOURS = [23, 0, 1, 2, 3]
        self.SOLAR_HOURS = [11, 12, 13, 14, 15, 16]

    def extract_features(self, df):
        df = df.copy()
        df['mag'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
        df['vedba'] = np.abs(df['mag'] - df['mag'].rolling(window=50, center=True).mean())
        return df.fillna(0)

    def predict_activity(self, df):
        conditions = [(df['vedba'] > self.RES_THRESHOLD), (df['vedba'] > 0.15)]
        df['activity_class'] = np.select(conditions, ['RES', 'FEED'], default='STANDING')
        return df

    def check_heat_stress(self, hourly_df):
        solar_data = hourly_df[hourly_df.index.hour.isin(self.SOLAR_HOURS)]
        if solar_data.empty: return False
        return solar_data['temp_mean'].mean() > 40.5

    def cattle_logic_engine(self, hourly_df, env_stress):
        night_data = hourly_df[hourly_df.index.hour.isin([0, 1, 2])]
        if night_data.empty:
            global_night_base = 0
        else:
            global_night_base = night_data['temp_mean'].min()

        daily_stats = []
        for day, day_data in hourly_df.groupby(hourly_df.index.date):
            night_subset = day_data[day_data.index.hour.isin(self.NIGHT_HOURS)]
            if night_subset.empty:
                night_max = global_night_base
            else:
                night_max = night_subset['temp_mean'].max()
            
            night_spike = night_max - global_night_base

            persistence = day_data['res_ratio'].rolling(3, min_periods=1).mean().max()

            score = (night_spike * 15) + (persistence * 40)
            print(f"DEBUG: Day {day} | Spike {night_spike} | Persist {persistence} | Score {score}")

            daily_stats.append({
                'date': str(day),
                'spike': max(0, night_spike),
                'res': persistence if not np.isnan(persistence) else 0,
                'score': score if not np.isnan(score) else 0,
                'stress': env_stress
            })

        daily_stats.sort(key=lambda x: x['score'], reverse=True)
        peak_date = daily_stats[0]['date'] if daily_stats else None
        return daily_stats, peak_date

    def generate_logs(self, results, peak_date):
        print("\n" + "="*80)
        print(f"INITIAL LOG: HEAT CYCLE DETECTED ON: {peak_date}")
        print(f"LOGIC: Sequential Pattern Match (Thermal Jump -> Behavioral Persistence)")
        print("="*80)

        results.sort(key=lambda x: x['date'])
        for r in results:
            if r['date'] == peak_date:
                status = "ALERT: CONFIRMED HEAT"
                if r['stress']: status = "SUPPRESSED (STRESS)"
            elif r['score'] > 25:
                status = "LOG: PROESTRUS"
            else:
                status = "NORMAL"

            print(f"[{r['date']}] {status:<22} | Night Spike: {r['spike']:.2f}C | Persistence: {round(r['res']*100)}%")

# --- PIPELINE EXECUTION ---
system = CattleIntelligenceSystem()
data_path = '124'
file_pattern = os.path.join(data_path, 'node-124-2026-*.csv')
files = sorted(glob.glob(file_pattern))

if not files:
    print("Error: No data files found.")
else:
    raw_data = pd.concat([pd.read_csv(f) for f in files])
    # Filter out invalid temperature 0 if necessary? 
    # Let's keep it as is to see what the user sees.
    raw_data['timestamp_ist'] = pd.to_datetime(raw_data['timestamp_ist'])

    features = system.extract_features(raw_data)
    ar_output = system.predict_activity(features)

    hourly = ar_output.set_index('timestamp_ist').resample('h').agg({'temperature_value': 'mean'})
    hourly.columns = ['temp_mean']
    # Calculate res_ratio
    res_flags = ar_output.set_index('timestamp_ist')['activity_class'].apply(lambda x: 1 if x == 'RES' else 0)
    hourly['res_ratio'] = res_flags.resample('h').mean()

    is_stress = system.check_heat_stress(hourly)
    results, peak_date = system.cattle_logic_engine(hourly, is_stress)
    system.generate_logs(results, peak_date)
