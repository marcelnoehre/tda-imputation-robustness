import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from src.constants import LABEL

def test_against_baseline(data, dependent_var, independent_var, baseline_var, groups):
    results = []
    variables = [v for v in data[independent_var].unique() if v != baseline_var]
    
    for var in variables:
        diffs = []

        for _, group in data.groupby(groups):
            base = group[group[independent_var] == baseline_var][dependent_var].dropna()
            comp = group[group[independent_var] == var][dependent_var].dropna()

            if not base.empty and not comp.empty:
                diffs.append(comp.median() - base.median())

        n_pairs = len(diffs)
        diffs = np.array(diffs)

        if n_pairs == 0:
            stat, p_value = np.nan, np.nan
            mean_diff = np.nan
        else:
            try:
                stat, p_value = wilcoxon(diffs)
            except ValueError:
                stat, p_value = np.nan, 1.0
            mean_diff = diffs.mean()

        results.append({
            'var': LABEL[var],
            'n_pairs': n_pairs,
            'mean_diff': mean_diff,
            'wilcoxon_stat': stat,
            'p_value': p_value
        })

    return pd.DataFrame(results)
