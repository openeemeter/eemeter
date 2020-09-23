import pandas as pd
import numpy as np
import os 
import pickle
import logging
import hashlib

def daily_load_shape(base, peak, t_up, t_down):
    if t_up < 0 or t_up > 23 or t_down < 0 or t_down > 23:
        raise ValueError("t_up and t_down must be between 0 and 24 inclusive.")
    df = pd.DataFrame({'t': range(24), 'value': base})
    if t_up < t_down:
        df.loc[(df.t > t_up) & (df.t <= t_down), 'value'] = peak
    else: 
        df.loc[(df.t > t_up) | (df.t <= t_down), 'value'] = peak  
    
    return df

def weekly_load_shape(daily_load_shape, noise_sigma=0.05):
    def rand():
        return np.random.lognormal(mean=0,sigma=noise_sigma, size=24)
    return pd.concat([pd.DataFrame({'t': f"{i}." + daily_load_shape.t.astype(str), 
                             'value': daily_load_shape.value * rand()}) for i in range(7)])
    

def cache(func, key, cache_folder='.cache'):
    path = os.path.join(cache_folder, hashlib.sha224(str(key).encode()).hexdigest())
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    if not os.path.exists(path):
        result = func()
        pickle.dump(result, open(path, 'wb'))
    return pickle.load(open(path, 'rb'))


class SyntheticMeter:
    """
    Compute synthetic data for one meter.  A very basic 24-hr load 
    shape is constructed which varies from a base value to a peak value and back, 
    with transition points at 9h and 20h, and random noise added.  
    (Transition points could be exposed as parameters in a future version.)  
    Three different load shapes are computed for winter months (Dec, Jan, Feb),
    summer months (June, July, Aug), and shoulder months (remainder).  

    Methods
    =======

    monthly(): 
        Compute a data frame of total monthly usage for each month.

    seasonal_168():
        Compute a data frame of weekly load shape (24*7 data points) for
        each of summer, winter, and shoulder season.  The same load shape
        is applied to each day; there is no weekend/weekday variation.

    features():
        Compute a data frame of features that summarize the meter.
        Currently available:
        - annual_usage: Total usage in one year
        - summer_usage: Total usage in the three summer months
        - winter_usage: Total usage in the three winter months

    Attributes
    ==========

    meter_id: str
        ID of the meter 
    winter_base: float
        Base usage in winter season.
    winter_peak: float
        Peak usage in winter season.
    summer_base: float
        Base usage in summer season.
    summer_peak: float
        Peak usage in summer season.
    shoulder_base: float
        Base usage in shoulder season.
    shoulder_peak: float
        Peak usage in shoulder season.
    noise_sigma: float
        Sigma term in zero-mean lognormal multiplicative noise.

    """    
    def __init__(self, meter_id, 
                winter_base, winter_peak, 
                summer_base, summer_peak, 
                shoulder_base, shoulder_peak,
                noise_sigma=0.05
                ):
        

        self.meter_id = meter_id
        self.winter_base = winter_base
        self.winter_peak = winter_peak
        self.summer_base = summer_base
        self.summer_peak = summer_peak
        self.shoulder_base = shoulder_base
        self.shoulder_peak = shoulder_peak
        self.noise_sigma = noise_sigma

        self.load_shape_winter = daily_load_shape(self.winter_base, self.winter_peak, t_up=9, t_down=20)
        self.load_shape_summer = daily_load_shape(self.summer_base, self.summer_peak, t_up=9, t_down=20)
        self.load_shape_shoulder = daily_load_shape(self.shoulder_base, self.shoulder_peak, t_up=9, t_down=20)
        
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

        winter_week = weekly_load_shape(self.load_shape_winter, noise_sigma=noise_sigma).assign(season='winter')
        shoulder_week = weekly_load_shape(self.load_shape_shoulder, noise_sigma=noise_sigma).assign(season='shoulder')
        summer_week = weekly_load_shape(self.load_shape_summer, noise_sigma=noise_sigma).assign(season='summer')

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
    """
    Compute synthetic data for a population of meters.  See `SyntheticMeter`.
    Data will be cached on disk in the folder specified in `cache_folder`.

    Methods
    =======

    monthly(): 
        Compute a data frame of total monthly usage for each month, for each meter.

    seasonal_168():
        Compute a data frame of weekly load shape (24*7 data points) for
        each of summer, winter, and shoulder season, for each meter.

    features():
        Compute a data frame of features that summarize the meter, for each meter.
        Currently available:
        - annual_usage: Total usage in one year
        - summer_usage: Total usage in the three summer months
        - winter_usage: Total usage in the three winter months

    Attributes
    ==========

    n_meters: int
        Number of meters to include in the population
    id_predix: str
        String value to be prefixed to each meter ID.
    winter_base_mean: float
        Mean term in lognormal random variable representing base usage in winter season.
    winter_base_sigma: float
        Sigma term in lognormal random variable representing base usage in winter season.
    winter_peak_mean: float
        Mean term in lognormal random variable representing peak usage in winter season.
    winter_peak_sigma: float
        Sigma term in lognormal random variable representing peak usage in winter season.
    summer_base_mean: float
        Mean term in lognormal random variable representing base usage in summer season.
    summer_base_sigma: float
        Sigma term in lognormal random variable representing base usage in summer season.
    summer_peak_mean: float
        Mean term in lognormal random variable representing peak usage in summer season.
    summer_peak_sigma: float
        Sigma term in lognormal random variable representing peak usage in summer season.
    shoulder_base_mean: float
        Mean term in lognormal random variable representing base usage in shoulder season.
    shoulder_base_sigma: float
        Sigma term in lognormal random variable representing base usage in shoulder season.
    shoulder_peak_mean: float
        Mean term in lognormal random variable representing peak usage in shoulder season.
    shoulder_peak_sigma: float
        Sigma term in lognormal random variable representing peak usage in shoulder season.
    cache_folder: str
        Folder in which to cache computed values.

    """    
    def __init__(self, n_meters, id_prefix,
                winter_base_mean, winter_base_sigma, 
                winter_peak_mean, winter_peak_sigma,
                summer_base_mean, summer_base_sigma, 
                summer_peak_mean, summer_peak_sigma,
                shoulder_base_mean, shoulder_base_sigma, 
                shoulder_peak_mean, shoulder_peak_sigma,
                cache_folder = '.cache'
                ):
        


        self.key = "_".join(f"{x}" for x in [n_meters, id_prefix, 
            winter_base_mean, winter_base_sigma, 
            winter_peak_mean, winter_peak_sigma, 
            summer_base_mean, summer_base_sigma, 
            summer_peak_mean, summer_peak_sigma,
            shoulder_base_mean, shoulder_base_sigma, 
            shoulder_peak_mean, shoulder_peak_sigma])

        self.n_meters = n_meters
        self.id_prefix = id_prefix
      
        self.df_params = pd.DataFrame({
            'meter_id': [f"{self.id_prefix}_{i}" for i in range(self.n_meters)],
            'winter_base': np.random.lognormal(mean=winter_base_mean, sigma=winter_base_sigma, size=n_meters),
            'winter_peak': np.random.lognormal(mean=winter_peak_mean, sigma=winter_peak_sigma, size=n_meters),
            'summer_base': np.random.lognormal(mean=summer_base_mean, sigma=summer_base_sigma, size=n_meters),
            'summer_peak': np.random.lognormal(mean=summer_peak_mean, sigma=summer_peak_sigma, size=n_meters),
            'shoulder_base': np.random.lognormal(mean=shoulder_base_mean, sigma=shoulder_base_sigma, size=n_meters),
            'shoulder_peak': np.random.lognormal(mean=shoulder_peak_mean, sigma=shoulder_peak_sigma, size=n_meters),

        })

        self.meters = None        
        
    def generate_meters(self):
        self.meters = [SyntheticMeter(
                meter_id = row['meter_id'],
                winter_base = row['winter_base'],
                winter_peak = row['winter_peak'],
                summer_base = row['summer_base'],
                summer_peak = row['summer_peak'],
                shoulder_base = row['shoulder_base'],
                shoulder_peak = row['shoulder_peak'])
            for ix, row in self.df_params.iterrows()]

    def monthly(self):
        def generate():
            if self.meters is None:
                self.generate_meters()
            return pd.concat([m.monthly() for m in self.meters])
        return cache(generate, self.key + 'monthly')

    def seasonal_168(self):
        def generate():
            if self.meters is None:
                self.generate_meters()
            return pd.concat([m.seasonal_168() for m in self.meters])
        return cache(generate, self.key + 'seasonal_168')

    def features(self):
        def generate():
            if self.meters is None:
                self.generate_meters()
            return pd.concat([m.features() for m in self.meters])
        return cache(generate, self.key + 'features')


class SyntheticTreatmentPoolPopulation:
    """
    Compute synthetic data for a treatment population and 
    comparison pool.  See `SyntheticMeter` and `SyntheticPopulation`.
    Data will be cached on disk in the folder specified in `cache_folder`.

   
    Methods
    =======

    monthly(): 
        Compute a data frame of total monthly usage for each month, for each meter.

    seasonal_168():
        Compute a data frame of weekly load shape (24*7 data points) for
        each of summer, winter, and shoulder season, for each meter.

    features():
        Compute a data frame of features that summarize the meter, for each meter.
        Currently available:
        - annual_usage: Total usage in one year
        - summer_usage: Total usage in the three summer months
        - winter_usage: Total usage in the three winter months

    Attributes
    ==========

    n_meters: int
        Number of meters to include in the population
    id_predix: str
        String value to be prefixed to each meter ID.
    winter_base_mean: float
        Mean term in lognormal random variable representing base usage in winter season.
    winter_base_sigma: float
        Sigma term in lognormal random variable representing base usage in winter season.
    winter_peak_mean: float
        Mean term in lognormal random variable representing peak usage in winter season.
    winter_peak_sigma: float
        Sigma term in lognormal random variable representing peak usage in winter season.
    summer_base_mean: float
        Mean term in lognormal random variable representing base usage in summer season.
    summer_base_sigma: float
        Sigma term in lognormal random variable representing base usage in summer season.
    summer_peak_mean: float
        Mean term in lognormal random variable representing peak usage in summer season.
    summer_peak_sigma: float
        Sigma term in lognormal random variable representing peak usage in summer season.
    shoulder_base_mean: float
        Mean term in lognormal random variable representing base usage in shoulder season.
    shoulder_base_sigma: float
        Sigma term in lognormal random variable representing base usage in shoulder season.
    shoulder_peak_mean: float
        Mean term in lognormal random variable representing peak usage in shoulder season.
    shoulder_peak_sigma: float
        Sigma term in lognormal random variable representing peak usage in shoulder season.
    cache_folder: str
        Folder in which to cache computed values.

    monthly(): 
        Compute a data frame of total monthly usage for each month, for each meter, 
        with a column `set` having value of either `treatment` or `pool`.

    seasonal_168():
        Compute a data frame of weekly load shape (24*7 data points) for
        each of summer, winter, and shoulder season, for each meter, 
        with a column `set` having value of either `treatment` or `pool`.

    features():
        Compute a data frame of features that summarize the meter, for each meter, 
        with a column `set` having value of either `treatment` or `pool`.
        Currently available:
        - annual_usage: Total usage in one year
        - summer_usage: Total usage in the three summer months
        - winter_usage: Total usage the three winter months

    Attributes
    ==========

    n_meters: int
        Number of meters to include in the population
    id_predix: str
        String value to be prefixed to each meter ID.
    treatment_winter_base_mean: float
        Mean term in lognormal random variable representing base usage in winter season, for treatment meters.
    treatment_winter_base_sigma: float
        Sigma term in lognormal random variable representing base usage in winter season, for treatment meters.
    treatment_winter_peak_mean: float
        Mean term in lognormal random variable representing peak usage in winter season, for treatment meters.
    treatment_winter_peak_sigma: float
        Sigma term in lognormal random variable representing peak usage in winter season, for treatment meters.
    treatment_summer_base_mean: float
        Mean term in lognormal random variable representing base usage in summer season, for treatment meters.
    treatment_summer_base_sigma: float
        Sigma term in lognormal random variable representing base usage in summer season, for treatment meters.
    treatment_summer_peak_mean: float
        Mean term in lognormal random variable representing peak usage in summer season, for treatment meters.
    treatment_summer_peak_sigma: float
        Sigma term in lognormal random variable representing peak usage in summer season, for treatment meters.
    treatment_shoulder_base_mean: float
        Mean term in lognormal random variable representing base usage in shoulder season, for treatment meters.
    treatment_shoulder_base_sigma: float
        Sigma term in lognormal random variable representing base usage in shoulder season, for treatment meters.
    treatment_shoulder_peak_mean: float
        Mean term in lognormal random variable representing peak usage in shoulder season, for treatment meters.
    treatment_shoulder_peak_sigma: float
        Sigma term in lognormal random variable representing peak usage in shoulder season, for treatment meters.
    pool_winter_base_mean: float
        Mean term in lognormal random variable representing base usage in winter season, for pool meters.
    pool_winter_base_sigma: float
        Sigma term in lognormal random variable representing base usage in winter season, for pool meters.
    pool_winter_peak_mean: float
        Mean term in lognormal random variable representing peak usage in winter season, for pool meters.
    pool_winter_peak_sigma: float
        Sigma term in lognormal random variable representing peak usage in winter season, for pool meters.
    pool_summer_base_mean: float
        Mean term in lognormal random variable representing base usage in summer season, for pool meters.
    pool_summer_base_sigma: float
        Sigma term in lognormal random variable representing base usage in summer season, for pool meters.
    pool_summer_peak_mean: float
        Mean term in lognormal random variable representing peak usage in summer season, for pool meters.
    pool_summer_peak_sigma: float
        Sigma term in lognormal random variable representing peak usage in summer season, for pool meters.
    pool_shoulder_base_mean: float
        Mean term in lognormal random variable representing base usage in shoulder season, for pool meters.
    pool_shoulder_base_sigma: float
        Sigma term in lognormal random variable representing base usage in shoulder season, for pool meters.
    pool_shoulder_peak_mean: float
        Mean term in lognormal random variable representing peak usage in shoulder season, for pool meters.
    pool_shoulder_peak_sigma: float
        Sigma term in lognormal random variable representing peak usage in shoulder season, for pool meters.
    cache_folder: str
        Folder in which to cache computed values.
    """

    def __init__(self, 
                n_treatment=100, n_pool=1000,
                treatment_winter_base_mean=0.1, treatment_winter_base_sigma=0.1, 
                treatment_winter_peak_mean=1, treatment_winter_peak_sigma=0.1, 
                treatment_summer_base_mean=0.1, treatment_summer_base_sigma=0.1, 
                treatment_summer_peak_mean=0.8, treatment_summer_peak_sigma=0.1, 
                treatment_shoulder_base_mean=0.1, treatment_shoulder_base_sigma=0.1, 
                treatment_shoulder_peak_mean=0.9, treatment_shoulder_peak_sigma=0.1, 
                pool_winter_base_mean=0.1, pool_winter_base_sigma=0.1, 
                pool_winter_peak_mean=1, pool_winter_peak_sigma=0.1, 
                pool_summer_base_mean=0.1, pool_summer_base_sigma=0.1, 
                pool_summer_peak_mean=0.8, pool_summer_peak_sigma=0.1, 
                pool_shoulder_base_mean=0.1, pool_shoulder_base_sigma=0.1, 
                pool_shoulder_peak_mean=0.9, pool_shoulder_peak_sigma=0.1, 
                cache_folder = '.cache'    
                ):

        logging.info(f"Caching objects to {cache_folder}")
        
        self.population_treatment = SyntheticPopulation(n_meters=n_treatment, id_prefix='treatment',
                winter_base_mean=treatment_winter_base_mean, 
                winter_base_sigma=treatment_winter_base_sigma, 
                winter_peak_mean=treatment_winter_peak_mean,
                winter_peak_sigma=treatment_winter_peak_sigma,
                summer_base_mean=treatment_summer_base_mean,
                summer_base_sigma=treatment_summer_base_sigma, 
                summer_peak_mean=treatment_summer_peak_mean,
                summer_peak_sigma=treatment_summer_peak_sigma,
                shoulder_base_mean=treatment_shoulder_base_mean,
                shoulder_base_sigma=treatment_shoulder_base_sigma, 
                shoulder_peak_mean=treatment_shoulder_peak_mean,
                shoulder_peak_sigma=treatment_shoulder_peak_sigma,
                cache_folder = cache_folder)
        
        self.population_pool = SyntheticPopulation(n_meters=n_pool, id_prefix='pool',
                winter_base_mean=pool_winter_base_mean, 
                winter_base_sigma=pool_winter_base_sigma, 
                winter_peak_mean=pool_winter_peak_mean,
                winter_peak_sigma=pool_winter_peak_sigma,
                summer_base_mean=pool_summer_base_mean,
                summer_base_sigma=pool_summer_base_sigma, 
                summer_peak_mean=pool_summer_peak_mean,
                summer_peak_sigma=pool_summer_peak_sigma,
                shoulder_base_mean=pool_shoulder_base_mean,
                shoulder_base_sigma=pool_shoulder_base_sigma, 
                shoulder_peak_mean=pool_shoulder_peak_mean,
                shoulder_peak_sigma=pool_shoulder_peak_sigma, 
                cache_folder = cache_folder)
        
        self.key = f"{self.population_pool.key}__{self.population_treatment.key}"

    def features(self):
        return pd.concat([
            self.population_treatment.features().assign(set='treatment'),
            self.population_pool.features().assign(set='pool'),
        ])
    
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


