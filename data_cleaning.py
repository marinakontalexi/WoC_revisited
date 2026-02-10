import pandas as pd

def remove_outliers(df, z_thresh=5):
    df = df.copy()
    outlier_mask = pd.Series(False, index=df.index)

    for (task, cond), group in df.groupby(['task_id', 'experimental_condition']):
        if cond != 'Control':
            continue  # only process control condition
        values = group['answer']
        if len(values) < 3:
            continue  # not enough data to compute outliers
        z_scores = (values - values.mean()) / values.std(ddof=0)
        outlier_mask.loc[group.index] = z_scores.abs() > z_thresh

    return df[~outlier_mask]

# Load the cleaned data from the pickle file
crowd = pd.read_pickle('../data/crowd.pkl')

crowd['answer'] = pd.to_numeric(crowd['answer'], errors='coerce')
crowd['correct_answer'] = pd.to_numeric(crowd['correct_answer'], errors='coerce')
crowd = crowd.dropna(subset=['answer', 'correct_answer'])     # Filter out tasks that have non-numeric answers

crowd = crowd.sort_values(by=['start_time', 'user_id'])
# for each experimental condition and task, assign an index based on the order of start_time
crowd['start_time_index'] = crowd.sort_values(by=['start_time', 'user_id']).groupby(['task_id', 'experimental_condition']).cumcount()

# task id to be int
crowd['task_id'] = crowd['task_id'].astype(int)
# spatial reasoning is a categorical domain
crowd = crowd[crowd['domain_name'] != 'spatial_reasoning']


crowd_no_outliers5 = remove_outliers(crowd, 5)
crowd_no_outliers6 = remove_outliers(crowd, 6)
crowd_no_outliers7 = remove_outliers(crowd, 7)

# Save the filtered data to a new pickle file
crowd.to_pickle('../data/crowd_full.pkl')
crowd_no_outliers5.to_pickle('../data/crowd_no_outliers5.pkl')
crowd_no_outliers6.to_pickle('../data/crowd_no_outliers6.pkl')
crowd_no_outliers7.to_pickle('../data/crowd_no_outliers7.pkl')