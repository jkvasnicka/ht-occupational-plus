'''
'''

import matplotlib.pyplot as plt

# FIXME: Should be able to load the function directly?
from raw_processing import osha_cleaning

#region: cumulative_changes
def cumulative_changes(log_file, initial_count):
    '''
    Plot the cumulative proportion of samples remaining after each cleaning 
    step.
    '''
    change_log = osha_cleaning.load_json(log_file)

    cumulative_pairs = prepare_cumulative_data(change_log, initial_count)
    steps = [step for step, _ in cumulative_pairs]
    cum_values = [count for _, count in cumulative_pairs]

    # Reverse the data to have the "full dataset" at the top
    steps = steps[::-1]
    cum_values = cum_values[::-1]

    fig, ax = plt.subplots(figsize=(6, 10))
    ax.plot(cum_values, steps, marker='o', linestyle='-')
    ax.set_title('Proportion of Samples Remaining After Each Cleaning Step')
    ax.set_xlabel('Samples Remaining (%)')
    ax.set_ylabel('Cleaning Step')
    ax.grid(True)
    
    # Invert the x-axis so that it decreases from left to right
    plt.gca().invert_xaxis()

    return fig, ax
#endregion

#region: prepare_cumulative_data
def prepare_cumulative_data(change_log, initial_count):
    '''
    Prepare the data for plotting the cumulative proportion of samples 
    remaining after each cleaning step.
    '''
    # Initialize with the full dataset
    cum_count = initial_count
    cumulative_pairs = [(f'0. Full dataset ({cum_count:,})', cum_count)]
    
    def reformat_key(k, step_number):
        return f"{step_number}. {k.capitalize().replace('_', ' ')}"
    
    for step_number, (k, v) in enumerate(change_log.items(), start=1):
        if abs(v) > 0:
            formatted_key = reformat_key(k, step_number)
            # Include the cumulative count of remaining samples
            cum_count += v  # where v = N_after - N_before
            cumulative_pairs.append((formatted_key, cum_count))

    # Convert the counts to proportions
    TO_PERCENT = 100
    cumulative_pairs = [
        (k, v/initial_count*TO_PERCENT) 
        for k, v in cumulative_pairs
        ]

    return cumulative_pairs
#endregion

#region: value_counts_hbar
def value_counts_hbar(counts, title, xlabel, ylabel):
    '''
    Creates a horizontal bar chart of value counts.

    Parameters
    ----------
    counts : pd.Series
        The return of pd.value_counts(), where the index are the categories
        and the values are the counts.
    '''
    # TODO: Consider adjusting size based on number of categories
    default_figsize = plt.rcParams['figure.figsize']
    fig, ax = plt.subplots(
        figsize=default_figsize[::-1]
    )

    counts_sorted = counts.sort_values()

    ax.barh(counts_sorted.index, counts_sorted.values, color='skyblue')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax
#endregion