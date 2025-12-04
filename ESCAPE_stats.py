import pingouin as pg
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

def pairwise_comparison(df, per_group_variables_scale, group_pairs = [("ED", "DZHI"), ("ED_CCS", "ED_AF"), ("DZHI_HCM", "DZHI_noHCM"), ("DZHI_priorCAD", "DZHI_nopriorCAD"), ("allcohorts_nopriorCAD", "allcohorts_priorCAD"), ("male", "female")], group_col="group", alpha=0.05, correction="fdr_bh"):
    variables = df.columns.drop(group_col)
    groups = df[group_col].unique()
    results = []
    for var in variables:
        for g1, g2 in group_pairs:
            if per_group_variables_scale[var]=="categorial":
                var_results = chi2_or_fisher_test(df.loc[df["group"].isin([g1, g2]), :], var, group_col)
            elif per_group_variables_scale[var]=="rational":
                var_results = mwu_or_ttest(df.loc[df["group"].isin([g1, g2]), :], var, group_col)
            elif per_group_variables_scale[var]=="free text":
                continue
            
            var_results["group_a"] = g1
            var_results["group_b"] = g2
            results.append(var_results)
            
    results_df = pd.DataFrame(results)
    original_pvals = results_df['p-value unadjusted']
    valid_mask = original_pvals.notna()
    valid_pvals = original_pvals[valid_mask]
    results_df['p-value adjusted'] = None
    results_df['significant'] = None
    if not valid_pvals.empty:
        reject, pvals_corr, _, _ = multipletests(valid_pvals, alpha=0.05, method='fdr_bh')
        results_df.loc[valid_mask, 'p-value adjusted'] = pvals_corr
        results_df.loc[valid_mask, 'significant'] = reject
    results_df['significance'] = results_df['p-value adjusted'].apply(lambda p: None if not p else '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.')
    return results_df

def chi2_or_fisher_test(df, variable, group_col):
    df_clean = df[[variable, group_col]].dropna()
    chi2_stat, p_chi2, dof, expected = chi2_contingency(pd.crosstab(df_clean[variable], df_clean[group_col]))

    p_value = None
    test_used = None

    if table.shape == (2, 2) and ((expected < 5).sum() > 0 or (expected < 1).sum() > 0):
        odds, p_value = fisher_exact(table)
        test_used = "Fisher's exact test"
    else:
        if (expected < 1).sum() > 0:
            pass
        elif (expected < 5).sum() > 0:
            p_value = p_chi2
            test_used = "Chi²-Test"
        else:
            p_value = p_chi2
            test_used = "Chi²-Test"

    return {
        "variable": variable, 
        'test': test_used,
        'p-value unadjusted': p_value,
    }

def mwu_or_ttest(df, variable, group_col, alpha=0.05):
    df_clean = df[[variable, group_col]].dropna()
    g1, g2 = df[group_col].unique()
    data1 = df[df[group_col] == g1][variable].dropna()
    data2 = df[df[group_col] == g2][variable].dropna()    
    
    n1 = len(data1)
    n2 = len(data2)
    if n1 < 3 or n2 < 3:
        return {
            'variable': variable,
            'group_a': g1,
            'group_b': g2,
            'test': None,
            "p-value unadjusted": None,
        }

    if data1.nunique() <= 1:
        p_shapiro1 = 0 
    else:
        p_shapiro1 = pg.normality(data1.to_frame(name=variable))['pval'].iloc[0]
    if data2.nunique() <= 1:
        p_shapiro2 = 0 
    else:
        p_shapiro2 = pg.normality(data2.to_frame())['pval'].iloc[0]
        
    if p_shapiro1 > alpha and p_shapiro2 > alpha:
        test = pg.ttest(data1, data2, correction=True)
        test_name = 't-test'
        pval = test['p-val'].values[0]
        stat = test['T'].values[0]
    else:
        test = pg.mwu(data1, data2)
        test_name = 'Mann-Whitney U'
        pval = test['p-val'].values[0]
        stat = test['U-val'].values[0]
    return {
        'variable': variable,
        'group_a': g1,
        'group_b': g2,
        'test': test_name,
        'p-value unadjusted': pval,
    }

format_p = lambda p, threshold=0.001: f"p < {threshold:.3f}" if p < threshold else f"p = {p:.3f}"
format_significance = lambda p: None if not p else '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

def build_flat_summary_dict(group_variable_comparison, variables_between_groups, per_group_variables_scale):
    summary = dict()
    grouped = group_variable_comparison.groupby('variable')
    for var, group_df in grouped:
        if type(var)==tuple:
            var, value = var
        else:
            value = " "
        if per_group_variables_scale[var]=="categorial":
            ns_dict = {}
            for _, row in group_df.iterrows():
                if row["n information available"] > 0:
                    ns_dict[(row['group'], "means ± std or observations/n (%)")] = f"{row['n']}/{row["n information available"]} ({row['n']/row["n information available"]:.2f})"
                    ns_dict[(row['group'], "available data")] = f"{row["n information available"]}/{row["group n"]} ({round(row["n information available"]/row["group n"], 2)})"
                else:
                    ns_dict[(row['group'], "means ± std or observations/n (%)")] = f"{row['n']}/{row["n information available"]} (-)"
                    ns_dict[(row['group'], "available data")] = f"{row["n information available"]}/{row["group n"]} (-)"
            sig_pairs = variables_between_groups[
                (variables_between_groups['variable'] == var) & 
                (variables_between_groups['significant'] == True)
            ]
            sig_list = [
                f"{row['group_a']} vs. {row['group_b']} ({row['significance']}, p-value: {format_p(row['p-value adjusted'])}, {row["test"]})"
                for _, row in sig_pairs.iterrows()
            ]
            sig_string = ', '.join(sig_list)
            ns_dict[('significant_comparisons', " ")] = sig_string
            summary[(var, value)] = ns_dict
        elif per_group_variables_scale[var]=="rational":
            means_dict = {}
            for _, row in group_df.iterrows():
                means_dict[(row['group'], "means ± std or observations/n (%)")] = f"{row['mean']:.2f}" + (f" ± {row['std']:.2f}" if not pd.isna(row['std']) else "")
                means_dict[(row['group'], "available data")] =  f"{row["n information available"]}/{row["group n"]} ({round(row["n information available"]/row["group n"], 2)})"
            sig_pairs = variables_between_groups[
                (variables_between_groups['variable'] == var) & 
                (variables_between_groups['significant'] == True)
            ]
            sig_list = [
                f"{row['group_a']} vs. {row['group_b']} ({row['significance']}, p-value: {format_p(row['p-value adjusted'])}, {row["test"]})"
                for _, row in sig_pairs.iterrows()
            ]
            sig_string = ', '.join(sig_list)
            means_dict[('significant_comparisons', " ")] = sig_string
            summary[(var, " ")] = means_dict
        elif per_group_variables_scale[var]=="free text":
            continue
    return summary


def comparisons_versus_reference(df_data, reference, pseudonyms_per_group, total_n, groups_for_reference = ["total", "ED", "DZHI", "ED_CCS"], median_references = ["hs troponin T [pg/ml]", "alanin-aminotransferase [U/l]"], name="reference"):
    variables_against_reference = {}
    for variable in reference["binary_and_continuous"].index:
        reference_value = reference["binary_and_continuous"].loc[variable, "mean_standard_unit"]
        if pd.notna(reference_value):
            n = reference["binary_and_continuous"].loc[variable, "n"]
            variables_against_reference[variable] = {name: {"means ± std or observations/n (%)": f"{round(reference_value, 2)} ± {round(reference["binary_and_continuous"].loc[variable, "SD_standard_unit"], 2)}", "available data": f"{n}/{total_n} ({round(n/total_n, 2)})"}}
            for group, pseudonyms in pseudonyms_per_group.items():
                if group in groups_for_reference:
                    df_ref = df_data.loc[df_data["pseudonym"].isin(pseudonyms), variable].dropna()
                    df_ref = df_ref.values
                    var_result = normality_or_onesided_test(df_ref, reference_value, variable, alpha=0.05, median_ref=variable in median_references)
                    variables_against_reference[variable][group] = {"means ± std or observations/n (%)": f"{round(df_ref.mean(), 2)} ± ({round(df_ref.std(), 2)})", "available data": f"{len(df_ref)}/{len(pseudonyms)} ({round(len(df_ref)/len(pseudonyms), 2)})", "p-value unadjusted": var_result["p-value unadjusted"], "test": var_result["test"]}
            pvals = [variables_against_reference[variable][g]["p-value unadjusted"] for g in groups_for_reference]
            reject, pvals_adj, _, _  = multipletests(pvals, alpha=0.05, method="fdr_bh")
            for i, g in enumerate (groups_for_reference):
                variables_against_reference[variable][g]["significance"] = format_significance(pvals_adj[i])
                variables_against_reference[variable][g]["p-value adjusted"] = format_p(pvals_adj[i])
        else:
            ref_pos = int(reference["binary_and_continuous"].loc[variable, "positive"])
            ref_neg = int(reference["binary_and_continuous"].loc[variable, "n"]-ref_pos)
            variables_against_reference[variable] = {name: {"means ± std or observations/n (%)": f"{ref_pos}/{ref_pos+ref_neg} ({round(ref_pos/(ref_pos+ref_neg), 2)})", "available data": f"{ref_pos+ref_neg}/{total_n} ({round((ref_pos+ref_neg)/total_n, 2)})"}}
            for group, pseudonyms in pseudonyms_per_group.items():
                if group in groups_for_reference:
                    if variable == "female sex":
                        data = df_data.loc[df_data["pseudonym"].isin(pseudonyms), "sex"].dropna()
                        data = data.replace({"male": 0, "female": 1})
                    elif variable == "dyspnea":
                        data = df_data.loc[df_data["pseudonym"].isin(pseudonyms), "dyspnoea as shortness of breath and/or trouble catching breath aggravated by physical exertion (ESC 2024)"].dropna()
                        data = data.replace({"yes": 1, "no": 0})
                    else:
                        data = df_data.loc[df_data["pseudonym"].isin(pseudonyms), variable].dropna()
                        data = data.replace({"yes": 1, "no": 0})
                        data = data.values
                    own_pos = sum(data)
                    own_neg = len(data) - own_pos
                
                    table = np.array([
                        [own_pos, own_neg],
                        [ref_pos, ref_neg]
                    ])
                    var_result = chi_square_or_fisher_for_reference(table, variable)
                    variables_against_reference[variable][group] = {"means ± std or observations/n (%)": f"{own_pos}/{len(data)} ({round(own_pos/len(data), 2)})", "available data": f"{len(data)}/{len(pseudonyms)} ({round(len(data)/len(pseudonyms), 2)})", "p-value unadjusted": var_result["p-value unadjusted"], "test": var_result["test"]}
            pvals = [variables_against_reference[variable][g]["p-value unadjusted"] for g in groups_for_reference]
            reject, pvals_adj, _, _  = multipletests(pvals, alpha=0.05, method="fdr_bh")
            for i, g in enumerate (groups_for_reference):
                variables_against_reference[variable][g]["significance"] = format_significance(pvals_adj[i])
                variables_against_reference[variable][g]["p-value adjusted"] = format_p(pvals_adj[i])
    return variables_against_reference

def chi_square_or_fisher_for_reference(table, variable):
    chi2_stat, p_chi2, dof, expected = chi2_contingency(table)

    p_value = None
    test_used = None
    
    if table.shape == (2, 2) and ((expected < 5).sum() > 0 or (expected < 1).sum() > 0):
        odds, p_value = fisher_exact(table)
        test_used = "Fisher's exact test"
    else:
        if (expected < 1).sum() > 0:
            pass
        elif (expected < 5).sum() > 0:
            p_value = p_chi2
            test_used = "Chi²-Test"
        else:
            p_value = p_chi2
            test_used = "Chi²-Test"

    return {
        "variable": variable, 
        'test': test_used,
        'p-value unadjusted': p_value,
    }

def normality_or_onesided_test(data, reference_value, variable, alpha=0.05, median_ref=False):
    n = len(data)
    if n < 3:
        return {
            'variable': variable,
            'test': None,
            'p-value unadjusted': None,
        }

    if len(np.unique(data)) <= 1:
        return {
            'variable': variable,
            'test': None,
            'p-value unadjusted': None,
        }

    _, p_shapiro = stats.shapiro(data)
    
    if (p_shapiro > alpha) & (not median_ref):
        t_stat, p_two_sided = stats.ttest_1samp(data, reference_value)
        
        # one-sided pvalue
        if data.mean() > reference_value:
            p_val = p_two_sided / 2 if t_stat > 0 else 1 - (p_two_sided / 2)
        else:
            p_val = p_two_sided / 2 if t_stat < 0 else 1 - (p_two_sided / 2)
        
        test_name = 't-test (one-sided)'
        stat_value = t_stat
        
    else:
        differences = data - reference_value
        non_zero_diffs = differences[differences != 0]
        
        if len(non_zero_diffs) == 0:
            return {
                'variable': variable,
                'test': None,
                'p-value unadjusted': None,
            }
        
        w_stat, p_val = stats.wilcoxon(non_zero_diffs, alternative="greater" if data.mean() > reference_value else "less")
        test_name = 'Wilcoxon (one-sided)'
        stat_value = w_stat
    
    return {
        'variable': variable,
        'test': test_name,
        'p-value unadjusted': p_val,
    }

def build_flat_summary_reference_dict(variables_against_reference, name):
    summary_ref_comparison = {}
    for variable in variables_against_reference:
        summary_ref_comparison[variable] = {}
        significant_comparisons_per_variable = []
        for group in variables_against_reference[variable].keys():
            summary_ref_comparison[variable][(group, 'means ± std or observations/n (%)')] = variables_against_reference[variable][group]['means ± std or observations/n (%)']
            summary_ref_comparison[variable][(group, 'available data')] = variables_against_reference[variable][group]["available data"]
            if group != name:
                if variables_against_reference[variable][group]["significance"] != "n.s.":
                    significant_comparisons_per_variable.append(f"{group} ({variables_against_reference[variable][group]['significance']}, p-value: {variables_against_reference[variable][group]["p-value adjusted"]}, {variables_against_reference[variable][group]["test"]})")
        summary_ref_comparison[variable][(f"significant comparisons with {name}", "")] = (", ").join(significant_comparisons_per_variable)
    return summary_ref_comparison