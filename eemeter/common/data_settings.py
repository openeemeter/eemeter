SufficiencyRequirements = {}

class Settings:
    def __init__(self):
        pass

class DailySettings(Settings):
    def __init__(self, n_days_kept_min : int = 350, cvrmse_adj_max : float = 0.3):
        # TODO : Reuse the daily settings for Caltrack at eemeter/eemeter/caltrack/daily/utilities/config.py
        self.n_days_kept_min = n_days_kept_min
        self.cvrmse_adj_max = cvrmse_adj_max

class MonthlySettings(Settings):
    def __init__(self, n_months_kept_min : int = 12, cvrmse_adj_max : float = 0.3):
        self.n_months_kept_min = n_months_kept_min
        self.cvrmse_adj_max = cvrmse_adj_max

