'''
'''
import matplotlib.pyplot as plt
_default_figsize = plt.rcParams['figure.figsize']

def chemical_counts(counts, title, xlabel, ylabel):
    """
    Creates a horizontal bar chart of chemical counts.

    Args:
    counts (pd.Series): A pandas Series where the index are the categories (subsectors or sectors)
                        and the values are the counts of chemicals.
    title (str): The title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    """
    fig, ax = plt.subplots(
        figsize=_default_figsize[::-1] # Consider adjusting size based on number of categories
    )

    # Sort the counts for better visualization
    counts_sorted = counts.sort_values()

    # Create the bar chart
    ax.barh(counts_sorted.index, counts_sorted.values, color='skyblue')

    # Add titles and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Show the plot
    fig.tight_layout()

    return fig, ax
    