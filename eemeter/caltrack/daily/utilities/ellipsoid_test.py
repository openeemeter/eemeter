import numpy as np

from scipy.ndimage import median_filter
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar


def ellipsoid_intersection_test(mu_A, mu_B, cov_A, cov_B):
    # Fix if all values are the same in 1 direction, "brent" doesn't work well with this
    if cov_A[1, 1] == 0:
        cov_A[1, 1] = 1E-14
    
    if cov_B[1, 1] == 0:
        cov_B[1, 1] = 1E-14

    lambdas, phi = eigh(cov_A, b=cov_B)
    v_squared = np.dot(phi.T, mu_A - mu_B) ** 2
           
    res = minimize_scalar(ellipsoid_K_function,
                        #   bracket = [0.0, 0.5, 1.0],
                          bounds = [0.0, 1.0],
                          args = (lambdas, v_squared),
                          method = "bounded")

    if res.fun[0] >= 0:
        return True
    return False


def ellipsoid_K_function(ss, lambdas, v_squared):
    ss = np.array(ss).reshape((-1,1))
    lambdas = np.array(lambdas).reshape((1,-1))
    v_squared = np.array(v_squared).reshape((1,-1))
    
    return 1 - np.sum(v_squared*((ss*(1 - ss))/(1 + ss*(lambdas - 1))), axis=1)


def confidence_ellipse(x, y, var=np.ones([2, 2])*1.96):
    # Applying a median filter to help with outliers
    idx_sorted = np.argsort(x).flatten()
    idx_original = np.argsort(idx_sorted).flatten()

    # size could be changed with justification
    y = median_filter(y[idx_sorted], size=5)[idx_original]

    # Computing the covariance and ellipse parameter values
    cov = np.cov(x, y)*var # scale covariances by std choice
    ab_sqr, v = np.linalg.eig(cov)
    [a, b] = np.sqrt(ab_sqr)
    phi = np.arctan2(*v[:, 0][::-1])

    mu = np.array([np.mean(x), np.mean(y)])

    return mu, cov, a, b, phi


def robust_confidence_ellipse(x, y, var=np.ones([2, 2])*1.96, outlier_std=3, N=2):
    var_outlier = np.ones([2, 2])*outlier_std**2

    # remove outliers in N iterations
    for n in range(N):
        mu, cov, a, b, phi = confidence_ellipse(x, y, var_outlier)

        # Center points
        xc = x - mu[0]
        yc = y - mu[1]

        # Rotate points so ellipse is aligned with axes
        phi *= -1
        xct = xc*np.cos(phi) - yc*np.sin(phi)
        yct = xc*np.sin(phi) + yc*np.cos(phi)

        # normalize to a circle of radius 1 
        r = (xct/a)**2 + (yct/b)**2

        idx = np.argwhere(r <= 1).flatten() # non-outlier points

        if len(x) - 3 <= len(idx):
            break

        x = x[idx]
        y = y[idx]

    if len(x) < 3:
        mu = cov = a = b = phi = None
        return [mu, cov, a, b, phi]
    
    return confidence_ellipse(x, y, var)


def ellipsoid_split_filter(meter, n_std=[1.4, 1.4]):
    if isinstance(n_std, float):
        var = np.ones([2, 2])*n_std**2
    else:
        std = np.array(n_std)[:, None]
        var = std.T*std

    cluster_ellipse = {}
    for season in ['summer', 'shoulder', 'winter']:
        for day_type, day_num in enumerate([[1, 2, 3, 4, 5], [6, 7]]):
            if day_type == 0:
                key = f"wd-{season[:2]}"
            else:
                key = f"we-{season[:2]}"

            meter_season = meter[(meter['season'] == season) & (meter['observed'].notna())]
            meter_season = meter_season[meter_season['day_of_week'].isin(day_num)]
            meter_season = meter_season.sort_values(by=['temperature'])

            T = meter_season['temperature'].values
            obs = meter_season['observed'].values

            if (len(T) < 3) or (len(obs) < 3):
                mu = cov = a = b = phi = None
            else:
                mu, cov, a, b, phi = robust_confidence_ellipse(T, obs, var)
                # mu, cov, a, b, phi = confidence_ellipse(T, obs, std_sqr)

            cluster_ellipse[key] = {"mu": mu, "cov": cov, "a": a, "b": b, "phi": phi}

    combos = {"summer": [[["wd-su", "wd-sh"], ["we-su", "we-sh"]], [["wd-su", "wd-wi"], ["we-su", "we-wi"]]],
              "shoulder": [[["wd-su", "wd-sh"], ["we-su", "we-sh"]], [["wd-sh", "wd-wi"], ["we-sh", "we-wi"]]],
              "winter": [[["wd-sh", "wd-wi"], ["we-sh", "we-wi"]], [["wd-su", "wd-wi"], ["we-su", "we-wi"]]],
              "weekday_weekend": [[["wd-su", "we-su"], ["wd-sh", "we-sh"], ["wd-wi", "we-wi"]]]}
    
    ellipse_overlap = {}
    allow_separate = {"summer": [False, False], "shoulder": [False, False], "winter": [False, False], "weekday_weekend": [False]}
    for key in allow_separate.keys():
        for i, season_wd_we in enumerate(combos[key]):
            for combo in season_wd_we:
                combo_str = "__".join(combo)

                if combo_str not in ellipse_overlap:
                    mu_A = cluster_ellipse[combo[0]]['mu']
                    cov_A = cluster_ellipse[combo[0]]['cov']
                    mu_B = cluster_ellipse[combo[1]]['mu']
                    cov_B = cluster_ellipse[combo[1]]['cov']

                    if all([coef is not None for coef in [mu_A, mu_B, cov_A, cov_B]]):
                        ellipse_overlap[combo_str] = ellipsoid_intersection_test(mu_A, mu_B, cov_A, cov_B)
                    else:
                        ellipse_overlap[combo_str] = False

                if not ellipse_overlap[combo_str]:
                    allow_separate[key][i] = True
                    break

        allow_separate[key] = all(allow_separate[key])

    return allow_separate