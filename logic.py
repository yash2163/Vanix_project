
class DailyAnalysisEngine:
    
    @staticmethod
    def extract_features_and_activity(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df["mag"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
        df["vedba"] = np.abs(df["mag"] - df["mag"].rolling(window=50, center=True).mean()).fillna(0)
        
        conditions = [
            df["vedba"] > Config.RES_THRESHOLD, 
            df["vedba"] > Config.FEED_THRESHOLD
        ]
        df["activity_class"] = np.select(conditions, ["RES", "FEED"], default="STANDING")
        return df

    @staticmethod
    def resample_to_hourly(df: pd.DataFrame, target_date: date) -> pd.DataFrame:
        """Convert raw per-second data into hourly buckets spanning precisely 24 hours."""
        if df.empty:
            full_index = pd.date_range(
                start=pd.Timestamp(datetime.combine(target_date, datetime.min.time())),
                end=pd.Timestamp(datetime.combine(target_date, datetime.max.time().replace(second=0))),
                freq="h"
            )
            empty_df = pd.DataFrame(index=full_index, columns=["temp_mean", "res_ratio"])
            return empty_df

        idx = df.set_index("timestamp_ist")
        hourly = idx["temperature_value"].resample("h").mean().to_frame(name="temp_mean")
        hourly["res_ratio"] = (
            idx["activity_class"]
            .apply(lambda x: 1 if x == "RES" else 0)
            .resample("h").mean()
        )
        
        # Enforce exact 24-hour index for the specific day
        full_index = pd.date_range(
            start=pd.Timestamp(datetime.combine(target_date, datetime.min.time())),
            end=pd.Timestamp(datetime.combine(target_date, datetime.max.time().replace(second=0))),
            freq="h"
        )
        hourly = hourly.reindex(full_index)
        return hourly

    @staticmethod
    def calculate_data_loss(df: pd.DataFrame) -> Tuple[float, dict]:
        """Calculate daily and per-hour data loss based on new DP counts."""
        if df.empty:
            return 100.0, {str(h): {"loss_pct": 100.0, "count": 0} for h in range(24)}

        # Ensure index is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp_ist"]):
            df["timestamp_ist"] = pd.to_datetime(df["timestamp_ist"])
            
        total_dps = len(df)
        daily_loss_pct = round(100.0 * (1 - (total_dps / Config.EXPECTED_DPS_PER_DAY)), 2)
        
        hourly_counts = df.groupby(df["timestamp_ist"].dt.hour).size().to_dict()
        hourly_data_dict = {}
        for h in range(24):
            count = int(hourly_counts.get(h, 0))
            loss_pct = round(100.0 * (1 - (count / Config.EXPECTED_DPS_PER_HOUR)), 2)
            # NO CLAMPING (allows negative loss for data gain)
            hourly_data_dict[str(h)] = {"loss_pct": loss_pct, "count": count}
            
        return daily_loss_pct, hourly_data_dict

    @staticmethod
    def calculate_daily_anchor(hourly_df: pd.DataFrame) -> float:
        """Find the minimum temperature between 11 PM and 4 AM (Isolated for this specific day)."""
        anchor_data = hourly_df[hourly_df.index.hour.isin(Config.ANCHOR_HOURS)]
        temps = anchor_data["temp_mean"].dropna()
        if temps.empty:
            return 0.0
        return float(temps.min())

    @staticmethod
    def calculate_metrics(hourly_df: pd.DataFrame, daily_anchor: float) -> dict:
        """Calculate spike, persistence, and score given an hourly dataframe and anchor."""
        night_data = hourly_df[hourly_df.index.hour.isin(Config.NIGHT_HOURS)]
        night_temps = night_data["temp_mean"].dropna()
        night_max = night_temps.max() if not night_temps.empty else np.nan
        
        spike = max(0.0, night_max - daily_anchor) if not np.isnan(night_max) else 0.0
        
        persistence = hourly_df["res_ratio"].rolling(3, min_periods=1).mean().max()
        persistence = 0.0 if np.isnan(persistence) else float(persistence)
        
        score = (spike * Config.SCORE_W_SPIKE) + (persistence * Config.SCORE_W_PERSIST)
        
        return {
            "daily_anchor_C": round(daily_anchor, 4),
            "night_spike_C": round(spike, 4),
            "persistence_pct": round(persistence * 100, 2),
            "score": round(score, 4)
        }