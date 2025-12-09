import numpy as np
import pandas as pd
import math 

from scipy.stats import norm
from sklearn import metrics, calibration
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.utils import resample
import numpy as np
import statsmodels.api as sm

def log_likelihood_from_probs(y, p):
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

def nagelkerke_coxsnell(y_true, y_pred_prob):
    # implementation based on the R package rcompanion v2.5.0 
    # Mangiafico, S. S. (2025). rcompanion: Functions to support extension education program evaluation [Manual]. Rutgers Cooperative Extension. https://CRAN.R-project.org/package=rcompanion
    y = np.asarray(y_true)
    p = np.asarray(y_pred_prob)
    n = len(y)
    ll_model = log_likelihood_from_probs(y, p)
    p0 = np.mean(y)
    ll_null = log_likelihood_from_probs(y, np.full(n, p0))

    r2_cs = 1.0 - np.exp((2.0 / n) * (ll_null - ll_model))

    denom = 1.0 - np.exp((2.0 / n) * ll_null)
    if np.isclose(denom, 0.0):
        r2_nk = np.nan
    else:
        r2_nk = r2_cs / denom

    return r2_cs, r2_nk, ll_model, ll_null

def calculate_net_benefit(y_true, y_pred_prob, thresholds, harm=0):
    n = len(y_true)
    net_benefits, tps, fps, predss = [], [], [], []

    for pt in thresholds:
        preds = (y_pred_prob >= pt).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        net_benefit = (tp / n) - (fp / n) * (pt / (1 - pt)) - harm
        net_benefits.append(net_benefit)  
        tps.append(tp/n)
        fps.append(fp/n)
        predss.append(preds.sum()/n)
    return net_benefits, tps, fps, predss


def calculate_net_benefit_negative(y_true, y_pred_prob, thresholds, harm=0):
    n = len(y_true)
    net_benefits_negative, tns, fns, predss = [], [], [], []

    for pt in thresholds:
        preds = (y_pred_prob >= pt).astype(int)
        tn = ((preds == 0) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        net_benefit_negative = (tn / n ) - (fn / n) * ((1-pt)/pt) - harm
        net_benefits_negative.append(net_benefit_negative)  
        tns.append(tn/n)
        fns.append(fn/n)
        predss.append((n-preds.sum())/n)
    return net_benefits_negative, tns, fns, predss

def smooth_array(x, y, smooth_frac=0.5):
    smooth_result = lowess(y, x, frac=smooth_frac, return_sorted=True)
    x_smooth = smooth_result[:, 0]
    y_smooth = smooth_result[:, 1]
    return x_smooth, y_smooth

def calculate_dca(y_true, y_pred_prob, model_name="model", pt_thresholds=np.arange(0.01, 1.0, 0.01), harm=0):
    prevalence = np.mean(y_true)
    
    net_benefit_model, tp, fp, pos = calculate_net_benefit(y_true, y_pred_prob, pt_thresholds, harm=harm)
    standardized_net_benefit_model = net_benefit_model/prevalence
    net_benefit_all = [(prevalence - (1 - prevalence) * (pt / (1 - pt))) for pt in pt_thresholds]
    net_benefit_none = [0 for _ in pt_thresholds]

    net_benefit_negative_model, tn, fn, neg = calculate_net_benefit_negative(y_true, y_pred_prob, pt_thresholds, harm=harm)
    standardized_net_benefit_negative_model = net_benefit_negative_model / (1 - prevalence)
    net_benefit_negative_all = [0 for _ in pt_thresholds]
    net_benefit_negative_none = [((1 - prevalence) - prevalence * ((1-pt)/pt)) for pt in pt_thresholds]
    
    dca_df = pd.DataFrame(
        {"model": [model_name]*len(pt_thresholds), 
         "threshold": pt_thresholds, 
         "n": [len(y_true)]*len(pt_thresholds), 
         "prevalence": [prevalence]*len(pt_thresholds), 
         
         "test_pos_rate": pos, 
         "tp_rate": tp, 
         "fp_rate": fp, 
         "net_benefit": net_benefit_model, 
         "standardized_net_benefit": standardized_net_benefit_model,
         "net_benefit_all": net_benefit_all,
         "net_benefit_none": net_benefit_none,

         "test_neg_rate": neg,
         "tn_rate": tn,
         "fn_rate": fn,
         "net_benefit_negative": net_benefit_negative_model,
         "standardized_net_benefit_negative": standardized_net_benefit_negative_model,
         "net_benefit_negative_all": net_benefit_negative_all,
         "net_benefit_negative_none": net_benefit_negative_none,
        }
    )
    return dca_df

def categorical_nri(old_probs, new_probs, y_true, thresholds):
    old_cat = pd.cut(old_probs, bins=thresholds, labels=False, right=False)
    new_cat = pd.cut(new_probs, bins=thresholds, labels=False, right=False)
    event_idx = y_true == 1
    nonevent_idx = y_true == 0
    n_event = np.sum(event_idx)
    n_nonevent = np.sum(nonevent_idx)
    up_event = np.sum(new_cat[event_idx] > old_cat[event_idx]) / n_event if n_event > 0 else np.nan
    down_event = np.sum(new_cat[event_idx] < old_cat[event_idx]) / n_event if n_event > 0 else np.nan
    up_nonevent = np.sum(new_cat[nonevent_idx] > old_cat[nonevent_idx]) / n_nonevent if n_nonevent > 0 else np.nan
    down_nonevent = np.sum(new_cat[nonevent_idx] < old_cat[nonevent_idx]) / n_nonevent if n_nonevent > 0 else np.nan
    nri_event = up_event - down_event
    nri_nonevent = down_nonevent - up_nonevent
    nri_total = nri_event + nri_nonevent
    return nri_total, nri_event, nri_nonevent

def bootstrap_nri(old_probs, new_probs, y_true, thresholds, n_bootstraps=1000, alpha=0.05):
    n = len(y_true)
    nri_values = []
    for _ in range(n_bootstraps):
        idxs = resample(np.arange(n), replace=True, n_samples=n)
        nri, _, _ = categorical_nri(old_probs[idxs], new_probs[idxs], y_true[idxs], thresholds)
        nri_values.append(nri)
    nri_values = np.array(nri_values)
    nri_point = categorical_nri(old_probs, new_probs, y_true, thresholds)[0]
    lower = np.percentile(nri_values, 100 * alpha/2)
    upper = np.percentile(nri_values, 100 * (1 - alpha/2))
    p_lower = np.mean(nri_values <= nri_point)
    p_upper = np.mean(nri_values >= nri_point)
    p_value = 2 * min(p_lower, p_upper)
    p_value = min(1, p_value)
    return nri_point, (lower, upper), p_value

def bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap=1000, ci=95):
    n = len(y_true)
    bootstrap_scores = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        try:
            score = metric_func(y_true_boot, y_pred_boot)
            if score is not None and score is not pd.NA:
                score_float = float(score)
                if np.isfinite(score_float):
                    bootstrap_scores.append(score_float)
        except (ValueError, ZeroDivisionError, TypeError):
            continue
            
    if len(bootstrap_scores) < 40:
        return "-", "-"
    
    bootstrap_scores = np.array(bootstrap_scores, dtype=float)
    point_estimate = metric_func(y_true, y_pred)
    
    alpha = (100 - ci) / 2
    lower = np.percentile(bootstrap_scores, alpha)
    upper = np.percentile(bootstrap_scores, 100 - alpha)
    
    return lower, upper

def calculate_discrimination_slope(y_true, score_values):
    mean_pos = score_values[y_true == 1].mean()
    mean_neg = score_values[y_true == 0].mean()
    discrimination_slope = mean_pos - mean_neg
    return discrimination_slope

def calculate_nri_with_threshold(df, model1, model2, outcome, threshold):
    old_cat = df[model1] >= threshold
    new_cat = df[model2] >= threshold
    diff = new_cat.astype(int) - old_cat.astype(int)
    df["diff"] = diff
    df_event = df[df[outcome] == 1]
    n_event = len(df_event)
    up_event = (df_event["diff"] == 1).sum()
    down_event = (df_event["diff"] == -1).sum()
    df_nonevent = df[df[outcome] == 0]
    n_nonevent = len(df_nonevent)
    up_nonevent = (df_nonevent["diff"] == 1).sum()
    down_nonevent = (df_nonevent["diff"] == -1).sum()
    p_up_event = up_event / n_event
    p_down_event = down_event / n_event
    p_up_nonevent = up_nonevent / n_nonevent
    p_down_nonevent = down_nonevent / n_nonevent
    nri_event = p_up_event - p_down_event
    nri_nonevent = p_down_nonevent - p_up_nonevent
    nri = nri_event + nri_nonevent
    var_event = (up_event + down_event) / (n_event**2) - ((up_event - down_event)**2) / (n_event**3)
    var_nonevent = (up_nonevent + down_nonevent) / (n_nonevent**2) - ((up_nonevent - down_nonevent)**2) / (n_nonevent**3)
    se = math.sqrt(var_event + var_nonevent)
    z = nri / se if se != 0 else np.nan
    p = 2 * norm.cdf(-abs(z))
    ci_low = nri - 1.96 * se
    ci_high = nri + 1.96 * se
    return pd.DataFrame({f"NRI at {threshold: .0%}": [ f"{round(nri, 2)}({round(ci_low, 2)}-{round(ci_high, 2)})", round(se, 4), round(z, 3), p]}, index=["index with 95 CI", "SE", "Z-score", "p-value"]).T
