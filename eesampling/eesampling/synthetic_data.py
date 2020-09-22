import pandas as pd
import numpy as np


def daily_load_shape(low, high, t_up, t_down):
    if t_up < 0 or t_up > 23 or t_down < 0 or t_down > 23:
        raise ValueError("t_up and t_down must be between 0 and 24 inclusive.")
    df = pd.DataFrame({'t': range(24), 'value': low})
    if t_up < t_down:
        df.loc[(df.t > t_up) & (df.t <= t_down), 'value'] = high
    else: 
        df.loc[(df.t > t_up) | (df.t <= t_down), 'value'] = high  
    
    return df

def weekly_load_shape(daily_load_shape, noise_sigma=0.05):
    def rand():
        return np.random.lognormal(mean=0,sigma=noise_sigma, size=24)
    return pd.concat([pd.DataFrame({'t': f"{i}." + daily_load_shape.t.astype(str), 
                             'value': daily_load_shape.value * rand()}) for i in range(7)])
    

class SyntheticMeter:
    def __init__(self, meter_id, 
                winter_low, winter_high, 
                summer_low, summer_high, 
                shoulder_low, shoulder_high,
                pct_noise=0.1
                ):
        self.meter_id = meter_id
        self.winter_low = winter_low
        self.winter_high = winter_high
        self.summer_low = summer_low
        self.summer_high = summer_high
        self.shoulder_low = shoulder_low
        self.shoulder_high = shoulder_high
        self.pct_noise = pct_noise

        self.load_shape_winter = daily_load_shape(self.winter_low, self.winter_high, t_up=9, t_down=20)
        self.load_shape_summer = daily_load_shape(self.summer_low, self.summer_high, t_up=9, t_down=20)
        self.load_shape_shoulder = daily_load_shape(self.shoulder_low, self.shoulder_high, t_up=9, t_down=20)
        
        self.winter_daily = self.load_shape_winter.value.sum()
        self.shoulder_daily = self.load_shape_shoulder.value.sum()
        self.summer_daily = self.load_shape_summer.value.sum()
        
    def monthly(self):
        def rand():
            return np.sum(np.random.lognormal(mean=0,sigma=0.5, size=30))
        return pd.DataFrame({'meter_id': self.meter_id, 
                            'month': np.arange(1,13), 
                            'value': [self.winter_daily*rand(), 
                                      self.winter_daily*rand(), 
                                      self.shoulder_daily*rand(), 
                                      self.shoulder_daily*rand(), 
                                      self.shoulder_daily*rand(), 
                                      self.summer_daily*rand(), 
                                      self.summer_daily*rand(), 
                                      self.summer_daily*rand(), 
                                      self.shoulder_daily*rand(), 
                                      self.shoulder_daily*rand(), 
                                      self.shoulder_daily*rand(), 
                                      self.winter_daily*rand()                                          
                                     ]})
        
    def seasonal_168(self):
        
        winter_week = weekly_load_shape(self.load_shape_winter, noise_sigma=0.05).assign(season='winter')
        shoulder_week = weekly_load_shape(self.load_shape_shoulder, noise_sigma=0.05).assign(season='shoulder')
        summer_week = weekly_load_shape(self.load_shape_summer, noise_sigma=0.05).assign(season='summer')
         
        df = pd.concat([winter_week, shoulder_week, summer_week])
        df['t']  = df['season'] + '.' + df['t'] 
        df = df[['t', 'value']]
        df['meter_id'] = self.meter_id
        return df.reset_index(drop=True)
        
    
    def features(self):
        df = self.monthly()
        winter_usage = df[df['month'] <= 3].value.sum()
        summer_usage = df[(df['month'] >= 6) & (df['month'] <= 8)].value.sum()
        annual_usage = df.value.sum()
        shoulder_usage =  - winter_usage - summer_usage
        return pd.DataFrame({'meter_id': self.meter_id, 'winter_usage': winter_usage, 
               'summer_usage': summer_usage, 'annual_usage': annual_usage}, index=[self.meter_id])
                                            
                                                                                            
class SyntheticPopulation:
    def __init__(self, n_meters, id_prefix,
                winter_low_mean, winter_low_sigma, 
                winter_high_mean, winter_high_sigma,
                summer_low_mean, summer_low_sigma, 
                summer_high_mean, summer_high_sigma,
                shoulder_low_mean, shoulder_low_sigma, 
                shoulder_high_mean, shoulder_high_sigma,
                 
                ):
        self.n_meters = n_meters
        self.id_prefix = id_prefix
      
        self.df_params = pd.DataFrame({
            'meter_id': [f"{self.id_prefix}_{i}" for i in range(self.n_meters)],
            'winter_low': np.random.lognormal(mean=winter_low_mean, sigma=winter_low_sigma, size=n_meters),
            'winter_high': np.random.lognormal(mean=winter_high_mean, sigma=winter_high_sigma, size=n_meters),
            'summer_low': np.random.lognormal(mean=summer_low_mean, sigma=summer_low_sigma, size=n_meters),
            'summer_high': np.random.lognormal(mean=summer_high_mean, sigma=summer_high_sigma, size=n_meters),
            'shoulder_low': np.random.lognormal(mean=shoulder_low_mean, sigma=shoulder_low_sigma, size=n_meters),
            'shoulder_high': np.random.lognormal(mean=shoulder_high_mean, sigma=shoulder_high_sigma, size=n_meters),

        })
        
        self.meters = [SyntheticMeter(
                meter_id = row['meter_id'],
                winter_low = row['winter_low'],
                winter_high = row['winter_high'],
                summer_low = row['summer_low'],
                summer_high = row['summer_high'],
                shoulder_low = row['shoulder_low'],
                shoulder_high = row['shoulder_high'])
            for ix, row in self.df_params.iterrows()]
            
    def monthly(self):
        return pd.concat([m.monthly() for m in self.meters])

    def seasonal_168(self):
        return pd.concat([m.seasonal_168() for m in self.meters])
        
    def features(self):
        return pd.concat([m.features() for m in self.meters])
        artist
        
        
class TreatmentPoolPopulation:
    def __init__(self, 
                n_treatment=100, n_pool=1000,
                treatment_winter_low_mean=0.1, treatment_winter_low_sigma=0.1, 
                treatment_winter_high_mean=1, treatment_winter_high_sigma=0.1, 
                treatment_summer_low_mean=0.1, treatment_summer_low_sigma=0.1, 
                treatment_summer_high_mean=0.8, treatment_summer_high_sigma=0.1, 
                treatment_shoulder_low_mean=0.1, treatment_shoulder_low_sigma=0.1, 
                treatment_shoulder_high_mean=0.9, treatment_shoulder_high_sigma=0.1, 
                pool_winter_low_mean=0.1, pool_winter_low_sigma=0.1, 
                pool_winter_high_mean=1, pool_winter_high_sigma=0.1, 
                pool_summer_low_mean=0.1, pool_summer_low_sigma=0.1, 
                pool_summer_high_mean=0.8, pool_summer_high_sigma=0.1, 
                pool_shoulder_low_mean=0.1, pool_shoulder_low_sigma=0.1, 
                pool_shoulder_high_mean=0.9, pool_shoulder_high_sigma=0.1            
                ):
        
        self.population_treatment = SyntheticPopulation(n_meters=n_treatment, id_prefix='treatment',
                winter_low_mean=treatment_winter_low_mean, 
                winter_low_sigma=treatment_winter_low_sigma, 
                winter_high_mean=treatment_winter_high_mean,
                winter_high_sigma=treatment_winter_high_sigma,
                summer_low_mean=treatment_summer_low_mean,
                summer_low_sigma=treatment_summer_low_sigma, 
                summer_high_mean=treatment_summer_high_mean,
                summer_high_sigma=treatment_summer_high_sigma,
                shoulder_low_mean=treatment_shoulder_low_mean,
                shoulder_low_sigma=treatment_shoulder_low_sigma, 
                shoulder_high_mean=treatment_shoulder_high_mean,
                shoulder_high_sigma=treatment_shoulder_high_sigma)
        
        self.population_pool = SyntheticPopulation(n_meters=n_pool, id_prefix='pool',
                winter_low_mean=pool_winter_low_mean, 
                winter_low_sigma=pool_winter_low_sigma, 
                winter_high_mean=pool_winter_high_mean,
                winter_high_sigma=pool_winter_high_sigma,
                summer_low_mean=pool_summer_low_mean,
                summer_low_sigma=pool_summer_low_sigma, 
                summer_high_mean=pool_summer_high_mean,
                summer_high_sigma=pool_summer_high_sigma,
                shoulder_low_mean=pool_shoulder_low_mean,
                shoulder_low_sigma=pool_shoulder_low_sigma, 
                shoulder_high_mean=pool_shoulder_high_mean,
                shoulder_high_sigma=pool_shoulder_high_sigma)
        
    def features_treatment(self):
        return self.population_treatment.features()
    
    def features_pool(self):
        return self.population_pool.features()
    
    def monthly(self):
        return pd.concat([
            self.population_treatment.monthly().assign(set='treatment'),
            self.population_pool.monthly().assign(set='pool'),            
        ])
    
    def seasonal_168(self):
        return pd.concat([
            self.population_treatment.seasonal_168().assign(set='treatment'),
            self.population_pool.seasonal_168().assign(set='pool'),            
        ])


