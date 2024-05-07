import pandas as pd


def estimate_q_mean(truth_matrix):
    joined = pd.merge(truth_matrix, left_on='task', right_on='task')
    q = joined.groupby('worker').apply(lambda x: (x['label_x'] == x['label_y']).mean())
    return q.to_list()