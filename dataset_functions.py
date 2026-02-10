import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, BoundaryNorm, to_hex
from matplotlib.gridspec import GridSpec
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scipy.stats import norm, gaussian_kde, beta, t, linregress, skew, skewtest, gmean
from scipy.stats import shapiro, kstest, anderson, ks_2samp, wasserstein_distance
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

import random
from tabulate import tabulate
import json, os, pickle
import heapq
import time
import re

from general_utils import *

conditions = {0: "Control", 1: "Consensus", 2: "Most recent", 3: "Most confident"}
conditions_idx = {key: i for i, key in conditions.items()}

palette = {'Consensus': 'hotpink', 'Control': 'gray', 'Most recent': 'orange', 'Most confident': 'olivedrab', 'Simulation' : 'royalblue'}

# load experiment data
df = pd.read_pickle('../data/crowd_full.pkl')

tasks = pd.read_csv('../experiment_data/tasks.csv.zip')  # Ensure the correct path
domains = pd.read_csv('../experiment_data/domains.csv.zip')

# simple get functions to retrieve data for a given task_id and condition
def get_prompt(task_id):
    result = tasks.loc[tasks['task_id'] == task_id, 'prompt']
    return result.iloc[0] if not result.empty else None  # Return first match or None if not found

def get_correct_answer(task_id):
    result = tasks.loc[tasks['task_id'] == task_id, 'correct_answer']
    return float(result.iloc[0]) if not result.empty else None  # Return the correct answer or None if not found

def get_answers(task_id, condition='Control', df=df):
    try:
        # get answers as an array
        return df[df['task_id'] == task_id].groupby(['experimental_condition'])['answer'].apply(list).to_dict()[condition]
    except Exception as e:
        print("Wrong task_id, condition or data type", e)
        return None
    
# build a dictionary with precomputed info for each task and condition
def build_task_params_df(df=df, c_tolerance=0, cutoff=5, normalization="perc"):
    """
    Precompute and store parameters for all tasks into a DataFrame.
    This avoids repeated calls to get_params() and normalization functions.
    """
    task_array = []
    all_tasks = df['task_id'].unique()
    dict_help = load_from_file('data/task_params_with_last_update_sims.json')

    for task_id in all_tasks:
        try:
            # Store essential values
            params = task_params(task_id, df=df, c_tolerance=c_tolerance)
            params['last_update_sims_domain'] = dict_help[str(task_id)]['last_update_sims_domain']
            params['last_update_sims_lambda_zero'] = dict_help[str(task_id)]['last_update_sims_lambda_zero']
            task_array.append(params)
        except Exception as e:
            print(f"Error computing parameters for task {task_id}: {e}")
            continue
    
    res = pd.DataFrame(task_array)
    res.set_index('task_id', inplace=True)
    return res


def task_params(task_id, df=df, c_tolerance=0, confidence=0.9):
    control = get_answers(task_id, condition='Control', df=df)
    control_meds = pd.Series(control).expanding().median().values

    consensus_ans = get_answers(task_id, condition='Consensus', df=df)
    consensus_meds = pd.Series(consensus_ans).expanding().median().values

    correct = get_correct_answer(task_id)

    # compute basic stats
    mu, sigma = norm.fit(control)
    std_dev = np.std(control, ddof=1)
    skewness = skew(control)
    
    if c_tolerance == 0:
        c = sum(consensus_ans[i] == consensus_meds[i-1] for i in range(3, len(consensus_ans))) / (len(consensus_ans) - 3)
    else:
        c = sum(abs(consensus_ans[i] - consensus_meds[i-1])/(consensus_meds[i-1]+0.01) <= c_tolerance for i in range(3, len(consensus_ans))) / (len(consensus_ans) - 3)

    k = min(1, c / (1 - c + 1e-5))

    # compute L, U, interval
    try:
        a, b, c1, c2 = find_ms(control, k, confidence=confidence)
    except Exception as e:
        print("Error in find_ms for task", task_id, e)
        raise e
    try:
        if c > 1/3:
            a_p, b_p, c1_p, c2_p = find_ms(control, 2*k-1, confidence=0.9)
        else:
            a_p, b_p, c1_p, c2_p = None, None, None, None
    except Exception as e:
        print("Error in find_ms for dominant interval for task", task_id, e)
        a_p, b_p, c1_p, c2_p = (None, None, None, None)
    
    ctrl_distances = sorted(-abs(np.array(control) - correct))
    normalizer_dist = normalize_func('perc', ctrl_distances)
    linear_normalizer = normalize_func('perc', control)

    scores = np.zeros(len(conditions))
    scores_avg = np.zeros(len(conditions))
    scores_gmean = np.zeros(len(conditions))
    scores_abs = np.zeros(len(conditions))
    absolute_error = np.zeros(len(conditions))
    last_upd = np.zeros(len(conditions))
    correct_perc = np.zeros(len(conditions))
    starting_accuracy = np.zeros(len(conditions))
    scores_mode = np.zeros(len(conditions))

    for idx, condition in conditions.items():
        ans = get_answers(task_id, condition=condition)
        meds = pd.Series(ans).expanding().median().values
        mode = np.mean(pd.Series(ans).mode()) if len(pd.Series(ans).mode()) > 0 else np.nan
        # compute scores for social conditions
        # compute starting point accuracy
        [scores[idx], 
         scores_avg[idx], 
         scores_gmean[idx], 
         starting_accuracy[idx],
         scores_mode[idx]] = percentile_score([meds[-1], 
                                               np.mean(ans), 
                                               gmean(np.array(ans)[np.array(ans) >= 0]), 
                                               np.median(ans[:3]),
                                               mode], correct, control)

        # compute scores with mse, mae
        scores_abs[idx] = (linear_normalizer(np.median(ans)) - linear_normalizer(correct))
        # compute last median updates
        last_upd[idx] = max((i for i in range(1,len(meds)) if meds[i] != meds[i-1]), default=0)
        # compute percentage of correct answers within 10% of the correct answer
        correct_perc[idx] = sum(abs(x - correct)/(correct + 1e-2) <= 0.001 for x in ans) / len(ans)
        # compute correct perc in terms of std dev
        correct_perc[idx] = sum(abs(x - correct)/(std_dev + 1e-2) <= 0.01 for x in ans) / len(ans)
        # compute absolute error
        absolute_error[idx] = abs(np.median(ans) - correct)


    return {'task_id' : task_id,
            "domain" : domains.loc[domains['domain_id'] == tasks.loc[tasks['task_id'] == task_id, 'domain_id'].values[0], 'domain_name'].values[0],
            "prompt": get_prompt(task_id),

            "cons_ans" : list(consensus_ans),
            "ctrl_ans" : list(control),
            "cons_meds" : list(consensus_meds),
            "ctrl_meds" : list(control_meds),

            "mean" : np.mean(control),
            "median" : np.median(control),
            "variance" : np.var(control),
            "std_dev" : std_dev,
            "mu" : mu, "sigma" : sigma,
            "skewness" : skewness,
            
            "scores" : list(scores),
            "scores_avg" : list(scores_avg),
            "scores_gmean" : list(scores_gmean),
            "scores_abs" : list(scores_abs),
            'scores_mode' : list(scores_mode),
            "starting_accuracy" : list(starting_accuracy),
            "last_median_update": list(last_upd),
            "perc_of_corrects" : list(correct_perc),
            "absolute_error" : list(absolute_error),

            # "kde" : None if not distr else construct_distribution_kde(control, show_distr, correct),

            "c" : c, "k" : k, 
            "final_median" : consensus_meds[-1],
            "correct" : correct,
            "correct_perc" : sum((x - correct)/(correct + 1e-2) <= 0.1 for x in control) / len(control),
            "median_distance" : abs(np.median(control) - correct),
            "mean_distance" : abs(np.mean(control) - correct),

            "L" : a, "U" : b,
            "conf_minus" : c1, "conf_plus" : c2,
            "dom_int": (a_p, b_p, c1_p, c2_p),
            }

def build_task_params(load=True):
    if load:
        try:
            task_params_df = pd.read_pickle("data/task_params_df.pkl")
            print("Loaded precomputed task_params_df from disk.")
            return task_params_df
        except Exception as e:
            print("Error loading precomputed task_params_df:", e)
    
    print("Building task_params_df from scratch...")
    task_params_df = build_task_params_df(df=df, c_tolerance=0)
    task_params_df.to_pickle("data/task_params_df.pkl")
    save_to_json(task_params_df.to_dict(orient='index'), "data/task_params")
    return task_params_df

def get_params(task_id, task_params_df=task_params_df):
    try:
        r = task_params_df.loc[task_id]
    except KeyError:
        raise ValueError(f"Task {task_id} not found in precomputed DataFrame.")
    return r.to_dict()

def compute_avg_c_per_domain(tolerance=0):
    mu_c_dict = {}
    ranges = {}
    for domain in df['domain_name'].unique():
        # print(f"Domain: {domain}")
        cs = 0
        max_c = 0
        min_c = 1
        for task in df[df['domain_name'] == domain]['task_id'].unique():
            params = task_params_dict[task]
            cs += params['c']
            max_c = max(max_c, params['c'])
            min_c = min(min_c, params['c'])
        avg_c = cs / len(df[df['domain_name'] == domain]['task_id'].unique())
        mu_c_dict[domain] = avg_c
        ranges[domain] = (avg_c - min_c, max_c - avg_c) 
    return mu_c_dict, ranges

c_tolerance = 0
task_params_df = build_task_params(load=True)
task_params_dict = task_params_df.to_dict(orient='index')
c_per_domain, _ = compute_avg_c_per_domain(tolerance=c_tolerance)
domains_ordered_by_c = sorted(c_per_domain.keys(), key=lambda x: c_per_domain[x])
ordered_domains = {dom: i+1 for i, dom in enumerate(domains_ordered_by_c)}

# plotting functions

def plot_answers_over_time(task_id, conds=conditions.keys(), df=df,
                           ans=True, avg=True, median=True,
                           ax=None):

    task_data = df[(df['task_id'] == task_id) & (df["experimental_condition"].isin([conditions[i] for i in conds]))].copy()

    # Compute mean and median over time for each condition
    task_data['avg'] = task_data['answer'].expanding().mean()
    task_data['median'] = task_data['answer'].expanding().median()

    if ax is None:
        f, ax = plt.subplots(figsize=(8, 4))

    # plot correct answer line
    correct_answer = task_data['correct_answer'].iloc[0]
    ax.axhline(y=correct_answer, color='red', linestyle='--', alpha=0.7)

    if ans:
        sns.lineplot(data=task_data, x='start_time_index', y='answer', 
                    hue='experimental_condition', palette=palette,
                    marker='o', markersize=4, alpha=0.7, ax=ax)
    if avg:
        sns.lineplot(data=task_data, x='start_time_index', y='avg', 
                     hue='experimental_condition', palette=palette,
                     linestyle='--', legend=False, ax=ax)
    if median:
        sns.lineplot(data=task_data, x='start_time_index', y='median', 
                     hue='experimental_condition', palette=palette,
                     linestyle=':', legend=False, ax=ax)
        
    ax.set_title(task_data['prompt'].iloc[0])
    ax.set_xlabel('Start Time')
    ax.set_ylabel('Answer')
    ax.legend(title='Condition')
    ax.grid(True)
    # plt.show()

def plot_domain_bar_groups(dict_groups, labels_groups, colors_groups=None, scales=None,
                           title="", path=None, ranges_groups=None, sorting=True):
    """
    Plots grouped bar charts for dictionaries over domains.
    
    Parameters:
    - dict_groups: List of lists of dictionaries. Each sublist is a group sharing a y-axis.
    - labels_groups: List of lists of labels corresponding to the dictionaries.
    - colors_groups: List of lists of colors (optional).
    - title: Plot title.
    """
    assert len(dict_groups) == len(labels_groups), "Groups and labels must match"

    if sorting:
        all_domains = sorted(set().union(*[d.keys() for group in dict_groups for d in group]))
    else:
        all_domains = list(set().union(*[d.keys() for group in dict_groups for d in group]))

    x = np.arange(len(all_domains))
    total_width = 0.8
    num_total_bars = sum(len(group) for group in dict_groups)
    bar_width = total_width / num_total_bars

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax = ax1
    ax_list = [ax]

    bar_pos = -total_width / 2
    legend_handles = []  # To store handles separately for each group
    legend_labels = []   # To store labels separately for each group

    for group_idx, (dicts, labels) in enumerate(zip(dict_groups, labels_groups)):
        # Use second y-axis if needed
        if group_idx > 0:
            ax = ax1.twinx()
            ax_list.append(ax)
            ax.spines['right'].set_position(('axes', 1 + 0.1 * (group_idx - 1)))
            ax.set_frame_on(True)

        group_handles = []  # To store handles for the current group
        group_labels = []   # To store labels for the current group

        for i, (d, label) in enumerate(zip(dicts, labels)):
            values = [d.get(domain, 0) for domain in all_domains]
            positions = x + bar_pos + i * bar_width
            color = colors_groups[group_idx][i] if colors_groups and group_idx < len(colors_groups) and i < len(colors_groups[group_idx]) else None
            bar = ax.bar(positions, values, bar_width, label=label, alpha=0.8, color=color)

            yerr_low = [ranges_groups[group_idx][i][domain][0] for domain in all_domains] if ranges_groups[group_idx] else None
            yerr_high = [ranges_groups[group_idx][i][domain][1] for domain in all_domains] if ranges_groups[group_idx] else None

            if yerr_high != None and yerr_high != None:
                yerr = [yerr_low, yerr_high]
                plt.errorbar(positions, values, yerr=yerr, fmt='o', color='black', capsize=1, elinewidth=1, capthick=1, markersize=3)

            # Store the bar handles and corresponding labels for later
            group_handles.append(bar)
            group_labels.append(label)

        # Append the group-specific handles and labels to the main list
        legend_handles.append(group_handles)
        legend_labels.append(group_labels)

        ax.set_yscale(scales[group_idx] if scales and group_idx < len(scales) else 'linear')
        
        bar_pos += len(dicts) * bar_width

    ax1.set_xticks(x)
    ax1.set_xticklabels(all_domains, rotation=45, ha='right')
    ax1.set_xlabel('Domain')
    ax1.grid(True, axis='y')
    fig.suptitle(title)
    
    # Combine all legends, making sure each group has its own legend
    for i, (group_handles, group_labels) in enumerate(zip(legend_handles, legend_labels)):
        ax_list[i].legend(group_handles, group_labels, loc='upper right' if i == 0 else 'upper left', bbox_to_anchor=(1, 1), frameon=False)

    plt.tight_layout()

    if path:
        plt.savefig(path, bbox_inches='tight')
    plt.show()

def influence_curve_task(task_id, saveto=None, nodups=True):
    params = task_params_dict[task_id]
    meds = get_answers(task_id, data='median')
    consensus_ans = sorted(params['cons_ans'])
    control_ans = sorted(params['ctrl_ans'])
    
    _, f = influence_curve_new(
        params['L'], params['U'],
        (params['conf_minus'], params['conf_plus']),
        eq=params['final_median'],
        # a_range=(params["min_ans"]-200, params["max_ans"]+200),
        x_range=(control_ans[5], control_ans[-5]),
        # a_range=(0,100),
        correct=params['correct'],
        correct_perc=params['correct_perc'],
        medians=get_answers(task_id, 'Consensus', 'median'),
        title=f"[{task_id}] {params['prompt']} {params['c']:.2f}",
        # control_medians=(min(meds), max(meds)),
        control_medians=(consensus_ans[5], consensus_ans[-5]), 
        median=params['median'],
        # points=[(params['cons_M_minus'], 0, "#D06425", 'Consensus M-'),
        #         (params['cons_M_plus'], 0, '#123456', 'Consensus M+')]
    )
    # print(f)
    if saveto:
        plt.savefig(saveto, dpi=300)
    else:
        plt.show()
    plt.close()

def influence_curve_domain(dom):
    for task_id in df[df['domain_name'] == dom]['task_id'].unique():
        # print(f"Task ID: {task_id} - Domain: {get_domain_name(task_id)}")

        influence_curve_task(task_id, saveto=f"./plots/{dom}/sic_{task_id}.")


# ----------- WoC PLOTTING FUNCTIONS -----------
def plot_avg_score_with_ci(alpha=0.95, title=None):
    """
    Plot average relative scores with parametric alpha confidence intervals.

    Args:
        alpha (float): significance level (e.g. 0.05 for 95% CI)
        title (str): plot title
    """
    plt.figure(figsize=(5, 4))
    plt.title('Average Score by Condition (relative to Control)')

    x = np.arange(len(conditions)-1)  # numeric positions
    colors = [palette[c] for c in conditions.values()][1:]

    score_list = []

    colors = (colormaps['Set2'](i) for i in range(len(conditions)-1))  # create an iterator
    offset = 0.05  # horizontal offset for the two score types
    
    for score_type in ['scores', 'scores_avg']:
        for task_id, params in task_params_dict.items():
            scores = np.array(params[score_type], dtype=float)
            scores = scores[1:] - scores[0]   # relative to control
            score_list.append(scores)

        score = np.vstack(score_list)
        n_tasks = len(score)

        # --- Compute means and standard errors ---
        final_score = np.mean(score, axis=0)
        std_error = np.std(score, axis=0, ddof=1) / np.sqrt(n_tasks)

        # --- Compute z critical value ---
        delta = 1 - alpha
        z_crit = norm.ppf(1 - delta / 2)

        # --- Compute confidence intervals ---
        conf_low = final_score - z_crit * std_error
        conf_high = final_score + z_crit * std_error

        # --- Plot ---

        # plt.bar(x, final_score, color=colors, width=0.1)
        plt.errorbar(x - offset if score_type=='scores' else x + offset, final_score, label='Median' if score_type=='scores' else 'Average',
                     yerr=[final_score - conf_low, conf_high - final_score],
                     fmt='o', color=next(colors), capsize=2, markerfacecolor='none', markersize=6)

        # for i, score in enumerate(final_score):
        #     plt.errorbar(i, score,
        #                 yerr=[[score - conf_low[i]], [conf_high[i] - score]],
        #                 fmt='o', color=colors[i], capsize=3, markerfacecolor='none', markersize=8)
        
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    plt.xticks(x, list(conditions.values())[1:])
    plt.ylabel('Average Relative Score')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'plots/WOC/simpleWOC.png', dpi=300)
    plt.show()

    # Return results for reuse
    return {
        "final_score": final_score,
        "conf_low": conf_low,
        "conf_high": conf_high,
        "z_crit": z_crit,
        "alpha": alpha
    }

def compute_stats_for_regression(
        func_x=lambda task_id: c_per_domain[task_params_dict[task_id]['domain']],
        func_y=lambda d: d['scores'][0],
        acc=(0, 1),
        condition=0,                # 0 for control, 1 for consensus, etc. (index into scores list)
        mae=True,                   # True -> MAE-like; False -> RMSE-like transform (square then sqrt for means)):
):
    X_all, Y_all = [], []

    X_means, Y_means = [], []
    mean_domain_ids = []         # 1..K labels for means (ordered)
    mean_domain_names = []       # domain strings aligned with means
    task_counts = []             # tasks per domain aligned with means
    mean_weights = []            # weights per domain mean (for weighted regression)

    domains_in_order = [d for d in ordered_domains.keys() if d != "spatial_reasoning"]

    eps = 1e-10
    for dom_idx, domain in enumerate(domains_in_order, start=1):
        task_ids = df.loc[df["domain_name"] == domain, "task_id"].unique()

        # per-task lists (for weights + optional label positioning)
        y_list = None
        n = 0

        for task_id in task_ids:
            if not (acc[0] < task_params_dict[task_id]["starting_accuracy"][condition] <= acc[1]):
                continue
            
            n += 1
            x = func_x(task_id)
            val = func_y(task_params_dict[task_id]) if mae else func_y(task_params_dict[task_id])**2

            X_all.append(x)
            Y_all.append(val if isinstance(val, (int, float)) else val[1] - val[0])  # if multiple conditions, take specified one for regression
            if y_list is None:
                y_list = np.array([val.copy()] if isinstance(val, np.ndarray) else [val])
            else:
                y_list = np.append(y_list, np.array([val.copy()]) if isinstance(val, np.ndarray) else np.array([val]), axis=0)

        if n == 0:
            continue

        # domain means
        domain_mean = np.sum(y_list, axis=0) / n if mae else np.sqrt(np.sum(y_list, axis=0) / n)
        y_mean = domain_mean if isinstance(domain_mean, (int, float)) else domain_mean[1] - domain_mean[0]  # if multiple conditions, take difference for regression
        X_means.append(func_x(task_ids[0]))  # all tasks in domain have same x, so just take first
        Y_means.append(y_mean)
        mean_domain_ids.append(dom_idx)
        mean_domain_names.append(domain)
        task_counts.append(n)

        # old weighting: n / var(list)
        if n <= 1:
            mean_weights.append(eps)
        else:
            v = float(np.var(y_list, ddof=1)) if isinstance(domain_mean, (int, float)) else np.var(y_list, axis=0, ddof=1)[1] + np.var(y_list, axis=0, ddof=1)[0]  # if multiple conditions, sum variances for weighting
            mean_weights.append(n / max(eps, v))

    return {
        "X_all": X_all,
        "Y_all": Y_all,
        "X_means": X_means,
        "Y_means": Y_means,
        "mean_domain_ids": mean_domain_ids,
        "mean_domain_names": mean_domain_names,
        "task_counts": task_counts,
        "mean_weights": mean_weights,
        }

def plot_tasks_vs_mu_c(
    *,
    func_x=lambda task_id: c_per_domain[task_params_dict[task_id]['domain']],
    func_y=lambda d: d['scores'][0],
    acc=(0, 1),
    condition=0,                # 0 for control, 1 for consensus, etc. (index into scores list)
    mae=True,                   # True -> MAE-like; False -> RMSE-like transform (square then sqrt for means)
    ax=None,
    color="#41156D",

    # labels and title
    title=None,
    label=None,
    label_alpha=0.7,
    label_fontsize=10,

    # axes labels
    ylabel=None,
    xlabel='copying probability',
    ylabel_fontsize=12,
    xlabel_fontsize=12,

    # display options
    show_tasks=True,
    alpha_task=0.25,
    s_task=14,

    # display means
    show_means=True,
    annotate_domain_means=True,   # put domain index text on means (1..K)
    alpha_mean=0.9,
    s_mean=25,
    color_mean=None,
    marker_mean="s",

    # regression
    fit_on="all",               # "all" | "means"
    show_fit=True,
    fit_label=True,
    joint_labels=False,

    # --- extra features ---
    weighted=False,             # weighted regression for fit_on="means"
    equaldots=True,             # if False, mean marker sizes scale with task_count
    show_zero_line=False,          # horizontal line at y=0 for difference plot

    plot_extra=False,           # plot extra series on domain means
    extra_y_per_domain=None,    # dict: {domain: value} (same x = mu_c_per_domain[domain])
    label_extra=None,
    color_extra="#2D78C3",
    marker_extra="x",

    return_data=False,          # return dict of arrays for downstream use
    grid_option='both',           # show axis grid
    alpha_grid=0.25,
):
    """
    x = mu_c(domain)

    y depends on `mode`:
      - "ctrl": control score
      - "cons": consensus score
      - "diff": consensus - control
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    def _weighted_linfit(x, y, w):
        """
        Weighted least squares line fit y ≈ a x + b
        Returns (a, b). (No p-value here.)
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.asarray(w, dtype=float)

        w = np.clip(w, 0.0, np.inf)
        if np.all(w == 0):
            w = np.ones_like(w)

        W = np.sum(w)
        xbar = np.sum(w * x) / W
        ybar = np.sum(w * y) / W

        denom = np.sum(w * (x - xbar) ** 2)
        if denom <= 0:
            return 0.0, ybar
        a = np.sum(w * (x - xbar) * (y - ybar)) / denom
        b = ybar - a * xbar
        return float(a), float(b)

    # ---------- collect ---------
    results = compute_stats_for_regression(
        func_x=func_x,
        func_y=func_y,
        acc=acc,
        condition=condition,
        mae=mae,
    )

    # nothing to plot
    if (not show_tasks or len(results["X_all"]) == 0) and (not show_means or len(results["X_means"]) == 0):
        print("No points to plot.")
        return ax

    # ---------- plot tasks ----------
    if show_tasks and len(results["X_all"]) > 0:
        ax.scatter(
            np.asarray(results["X_all"]), np.asarray(results["Y_all"]),
            s=s_task, alpha=alpha_task, color=color,
            label=label if not joint_labels else None
        )

    # ---------- plot means ----------
    if show_means and len(results["X_means"]) > 0:
        if equaldots:
            sizes = np.full(len(results["X_means"]), s_mean, dtype=float)
        else:
            sizes = s_mean/20 * np.asarray(results["task_counts"], dtype=float)

        ax.scatter(
            np.asarray(results["X_means"]), np.asarray(results["Y_means"]),
            s=sizes, marker=marker_mean, alpha=alpha_mean,
            color=make_color_darker(color, factor=0.7) if color_mean is None else color_mean,
            label=label if not (show_tasks or joint_labels) else None
        )

        # old labels_position: place ordered_domains[domain] at chosen y
        if annotate_domain_means:
            for x, domain, ym in zip(results["X_means"], results["mean_domain_names"], results["Y_means"]):
                y_text = ym

                label = ordered_domains[domain] if domain in ordered_domains else str(domain)
                ax.text(x, y_text, label, fontsize=label_fontsize, alpha=label_alpha, weight="bold")

    # baseline for diff
    if show_zero_line:
        ax.axhline(0, color="black", ls="--", lw=1)

    # ---------- plot extra (domain means overlay) ----------
    if plot_extra and (extra_y_per_domain is not None) and show_means and len(results["X_means"]) > 0:
        extra_x, extra_y = [], []
        for domain in results["mean_domain_names"]:
            if domain in extra_y_per_domain:
                extra_x.append(c_per_domain[domain])
                extra_y.append(extra_y_per_domain[domain])

        if len(extra_x) > 0:
            ax.plot(
                extra_x, extra_y,
                marker_extra, color=color_extra,
                label=(label_extra if label_extra is not None else "Extra"),
                markersize=6, alpha=0.85
            )

    # ---------- regression ----------
    if show_fit:
        if fit_on == "all":
            x_fit = np.asarray(results["X_all"])
            y_fit = np.asarray(results["Y_all"])
            w_fit = None
        elif fit_on == "means":
            x_fit = np.asarray(results["X_means"])
            y_fit = np.asarray(results["Y_means"])
            w_fit = np.asarray(results["mean_weights"])
        else:
            raise ValueError("fit_on must be one of: 'all', 'means'")

        if len(x_fit) >= 2:
            if weighted and (fit_on == "means"):
                slope, intercept = _weighted_linfit(x_fit, y_fit, w_fit)
                p = None
            else:
                slope, intercept, r, p, se = linregress(x_fit, y_fit)

            xx = np.linspace(x_fit.min(), x_fit.max(), 200)
            lab = f"{label}\n" if joint_labels else ""
            if fit_label:
                if p is None:
                    lab = lab + f"β={slope:.2f}" # (weighted, {fit_on})"
                else:
                    lab = lab + f"β={slope:.2f}, p={p:.3f}" # ({fit_on})"
            else:
                lab = None # f"Fit({fit_on})"

            ax.plot(xx, slope * xx + intercept, color=color, alpha=0.7, label=lab)
        
        else:
            print("Not enough points for regression fit.")

    ax.set_xlabel(xlabel if xlabel != "default" else "copying parameter", fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel if ylabel != "default" else "percentile score", fontsize=ylabel_fontsize)
    ax.set_title(title, fontsize=ylabel_fontsize+2)
    ax.grid(grid_option, axis=grid_option if grid_option else 'both', alpha=alpha_grid)
    ax.legend()

    if return_data:
        return {
            "X_all": np.asarray(results["X_all"]),
            "Y_all": np.asarray(results["Y_all"]),
            "X_means": np.asarray(results["X_means"]),
            "Y_means": np.asarray(results["Y_means"]),
            "domains": list(results["mean_domain_names"]),
            "task_counts": np.asarray(results["task_counts"]),
            "mean_weights": np.asarray(results["mean_weights"]),
            "fitting": {
                "slope": slope,
                "intercept": intercept,
                "p_value": p if 'p' in locals() else None,
                "r_squared": r**2 if 'r' in locals() else None,
            }
        }

    return ax


def plot_ctrl_cons_diff_vs_mu_c(
        func_y=lambda d: d['scores'],  # should return list/array of scores
        color1 =palette['Control'], color2=palette['Consensus'], color_diff="#8828AE",
        label1="Control", label2="Social influence", label_diff="Difference",
        ax=None, title=None,
        base=dict()):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(4, 4))

    base1 = {
        'func_y': lambda d: func_y(d)[0],
        'color': color1,
        'label': label1,
        'annotate_domain_means': False,
        'ax': ax,
        **base
    }
    ret1 = plot_tasks_vs_mu_c(**base1)

    base2 = {
        'func_y': lambda d: func_y(d)[1],
        'color': color2,
        'label': label2,
        'annotate_domain_means': False,
        'ax': ax,
        **base
    }
    ret2 = plot_tasks_vs_mu_c(**base2)

    base_diff = {
        'func_y': func_y,
        'ax': ax,
        'color': color_diff,
        'label': label_diff,
        'annotate_domain_means': True,
        'title': title,
        'show_zero_line': True,
        **base
    }
    ret3 = plot_tasks_vs_mu_c(**base_diff)

    return {
        "control": ret1,
        "consensus": ret2,
        "difference": ret3,
    }