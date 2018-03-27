import numpy as np
import statsmodels.formula.api as smf


def _fit_intercept(df, weighted=False):
    int_formula = 'upd ~ 1'
    try:
        if weighted:
            int_mod = smf.wls(formula=int_formula, data=df, weights=df['ndays'])
        else:
            int_mod = smf.ols(formula=int_formula, data=df)
        int_res = int_mod.fit()
    except:  # TODO: catch specific error
        int_rsquared, int_qualified = 0, False
        int_formula, int_mod, int_res = None, None, None
    else:
        int_rsquared, int_qualified = 0, True

    return int_formula, int_mod, int_res, int_rsquared, int_qualified


def _fit_cdd_only(df, weighted=False, billing=False):

    bps = [i[4:] for i in df.columns if i[:3] == 'CDD']
    best_bp, best_rsquared, best_mod, best_res = None, -9e9, None, None
    best_formula, cdd_qualified = None, False

    try:  # TODO: fix big try block anti-pattern
        for bp in bps:
            candidate_cdd_formula = 'upd ~ CDD_' + bp
            if not billing:
                if (np.nansum(df['CDD_' + bp] > 0) < 10) or \
                   (np.nansum(df['CDD_' + bp]) < 20):
                    continue
            if weighted:
                candidate_cdd_mod = smf.wls(formula=candidate_cdd_formula, data=df,
                                            weights=df['ndays'])
            else:
                candidate_cdd_mod = smf.ols(formula=candidate_cdd_formula, data=df)
            candidate_cdd_res = candidate_cdd_mod.fit()
            candidate_cdd_rsquared = candidate_cdd_res.rsquared_adj
            if (candidate_cdd_rsquared > best_rsquared and
                    candidate_cdd_res.params['Intercept'] >= 0 and
                    candidate_cdd_res.params['CDD_' + bp] >= 0 and
                    candidate_cdd_res.pvalues['CDD_' + bp] < 0.1):
                best_bp, best_rsquared = int(bp), candidate_cdd_rsquared
                best_mod, best_res = candidate_cdd_mod, candidate_cdd_res
                cdd_qualified = True
                best_formula = 'upd ~ CDD_' + bp
    except:  # TODO: catch specific error
        best_rsquared, cdd_qualified = 0, False
        best_formula, best_mod, best_res = None, None, None
        best_bp = None

    return best_formula, best_mod, best_res, best_rsquared, cdd_qualified, best_bp


def _fit_hdd_only(df, weighted=False, billing=False):

    bps = [i[4:] for i in df.columns if i[:3] == 'HDD']
    best_bp, best_rsquared, best_mod, best_res = None, -9e9, None, None
    best_formula, hdd_qualified = None, False

    try:  # TODO: fix big try block anti-pattern
        for bp in bps:
            candidate_hdd_formula = 'upd ~ HDD_' + bp
            if not billing:
                if (np.nansum(df['HDD_' + bp] > 0) < 10) or \
                   (np.nansum(df['HDD_' + bp]) < 20):
                    continue
            if weighted:
                candidate_hdd_mod = smf.wls(formula=candidate_hdd_formula, data=df,
                                            weights=df['ndays'])
            else:
                candidate_hdd_mod = smf.ols(formula=candidate_hdd_formula, data=df)
            candidate_hdd_res = candidate_hdd_mod.fit()
            candidate_hdd_rsquared = candidate_hdd_res.rsquared_adj
            if (candidate_hdd_rsquared > best_rsquared and
                    candidate_hdd_res.params['Intercept'] >= 0 and
                    candidate_hdd_res.params['HDD_' + bp] >= 0 and
                    candidate_hdd_res.pvalues['HDD_' + bp] < 0.1):
                best_bp, best_rsquared = int(bp), candidate_hdd_rsquared
                best_mod, best_res = candidate_hdd_mod, candidate_hdd_res
                hdd_qualified = True
                best_formula = 'upd ~ HDD_' + bp
    except:  # TODO: catch specific error
        best_rsquared, hdd_qualified = 0, False
        best_formula, best_mod, best_res = None, None, None
        best_bp = None

    return best_formula, best_mod, best_res, best_rsquared, hdd_qualified, best_bp


def _fit_full(df, weighted=False, billing=False):

    hdd_bps = [i[4:] for i in df.columns if i[:3] == 'HDD']
    cdd_bps = [i[4:] for i in df.columns if i[:3] == 'CDD']

    best_hdd_bp, best_cdd_bp, best_rsquared, best_mod, best_res = \
        None, None, -9e9, None, None
    best_formula, full_qualified = None, False

    try:  # TODO: fix big try block anti-pattern
        for hdd_bp in hdd_bps:
            for cdd_bp in cdd_bps:
                if cdd_bp < hdd_bp:
                    continue
                candidate_full_formula = 'upd ~ CDD_' + cdd_bp + \
                                         ' + HDD_' + hdd_bp
                if not billing:
                    if (np.nansum(df['HDD_' + hdd_bp] > 0) < 10) or \
                       (np.nansum(df['HDD_' + hdd_bp]) < 20):
                        continue
                    if (np.nansum(df['CDD_' + cdd_bp] > 0) < 10) or \
                       (np.nansum(df['CDD_' + cdd_bp]) < 20):
                        continue
                if weighted:
                    candidate_full_mod = smf.wls(formula=candidate_full_formula, data=df,
                                                 weights=df['ndays'])
                else:
                    candidate_full_mod = smf.ols(formula=candidate_full_formula, data=df)
                candidate_full_res = candidate_full_mod.fit()
                candidate_full_rsquared = candidate_full_res.rsquared_adj
                if (candidate_full_rsquared > best_rsquared and
                        candidate_full_res.params['Intercept'] >= 0 and
                        candidate_full_res.params['HDD_' + hdd_bp] >= 0 and
                        candidate_full_res.params['CDD_' + cdd_bp] >= 0 and
                        candidate_full_res.pvalues['HDD_' + hdd_bp] < 0.1 and
                        candidate_full_res.pvalues['CDD_' + cdd_bp] < 0.1):
                    best_hdd_bp, best_cdd_bp, best_rsquared = \
                        int(hdd_bp), int(cdd_bp), candidate_full_rsquared
                    best_mod, best_res = candidate_full_mod, candidate_full_res
                    full_qualified = True
                    best_formula = 'upd ~ CDD_' + cdd_bp + ' + HDD_' + hdd_bp
    except:  # TODO: catch specific error
        best_rsquared, full_qualified = 0, False
        best_formula, best_mod, best_res = None, None, None
        best_hdd_bp, best_hdd_bp = None, None

    return best_formula, best_mod, best_res, best_rsquared, full_qualified, best_hdd_bp, best_cdd_bp
