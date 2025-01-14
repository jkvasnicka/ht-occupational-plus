'''
'''

import numpy as np
from scipy.stats import pearsonr 

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib_venn import venn2
import seaborn as sns

# FIXME: Should be able to load the function directly?
from raw_processing import osha_cleaning

# TODO: Move to plot config
EC_LABEL = r'$\log_{10}(\mathit{EC})$ [mg$\cdot$(m$^3$)$^{-1}$]'

NOTE = 'Note: Zeros (non-detects) replaced with 0.5 x smallest non-zero value before log-transformation.'

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

    # Dynamically adjust figure height based on the number of steps
    fig_height = max(7, len(steps)*0.35)
    fig, ax = plt.subplots(figsize=(6, fig_height))
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
    n_in_parentheses = lambda cum_count : f'({cum_count:,})'
    first_label = '0. Full dataset ' + n_in_parentheses(cum_count)
    cumulative_pairs = [(first_label, cum_count)]
    
    def format_key(k, step_number):
        return f"{step_number}. {k.capitalize().replace('_', ' ')}"
    
    for step_number, (k, v) in enumerate(change_log.items(), start=1):
        if abs(v) > 0:
            formatted_key = format_key(k, step_number)
            # Include the cumulative count of remaining samples
            cum_count += v  # where v = N_after - N_before
            cumulative_pairs.append((formatted_key, cum_count))

    # Append the sample size to the last label
    last_pair = cumulative_pairs[-1]
    last_label = last_pair[0] + ' ' + n_in_parentheses(cum_count)
    cumulative_pairs[-1] = (last_label, last_pair[-1])

    # Convert the counts to proportions
    TO_PERCENT = 100
    cumulative_pairs = [
        (k, v/initial_count*TO_PERCENT) 
        for k, v in cumulative_pairs
        ]

    return cumulative_pairs
#endregion

# TODO: Consider adjusting size based on number of categories
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

#region: dataset_overlap_venn
def dataset_overlap_venn(
        usis_keys, 
        cehd_keys, 
        title=None, 
        write_path=None
        ):
    '''
    '''
    plt.figure()
    venn = venn2([usis_keys, cehd_keys], ('USIS', 'CEHD'))
    
    N_usis_only = len(usis_keys - cehd_keys)
    N_intersection = len(usis_keys & cehd_keys)
    N_cehd_only = len(cehd_keys - usis_keys)
    N_union = len(usis_keys | cehd_keys)
    
    usis_label = f'{N_usis_only:,} ({N_usis_only/N_union*100:.0f}%)'
    intersection_label = f'{N_intersection:,} ({N_intersection/N_union*100:.0f}%)'
    cehd_label = f'{N_cehd_only:,} ({N_cehd_only/N_union*100:.0f}%)'
    
    venn.get_label_by_id('10').set_text(usis_label)
    venn.get_label_by_id('11').set_text(intersection_label)
    venn.get_label_by_id('01').set_text(cehd_label)
    
    plt.title(title)

    if write_path:
        plt.gcf().savefig(write_path)
#endregion

#region: histogram_grid
def histogram_grid(targets_dict, write_path=None):
    '''
    Plots pooled histograms side-by-side for datasets (USIS, CEHD, Combined)
    and hierarchies (e.g., Sector, Subsector) from top to bottom.
    '''
    datasets = list(targets_dict.keys())  # ['USIS', 'CEHD', 'Combined']
    hierarchies = list(targets_dict[datasets[0]].keys())  # ['Sector', 'Subsector']
    n_rows = len(hierarchies)
    n_cols = len(datasets)

    # Compute global x-axis range
    all_data = []
    for dataset in datasets:
        for hierarchy in hierarchies:
            # Preprocess and collect data
            series = preprocess_target(targets_dict[dataset][hierarchy])
            all_data.extend(series)  # Append for global range calculation

    # Calculate consistent x-axis limits
    x_min, x_max = np.floor(min(all_data)), np.ceil(max(all_data))

    fig, axes = plt.subplots(
        n_rows, 
        n_cols, 
        figsize=(4 * n_cols, 3 * n_rows), 
        sharex=True, 
        sharey=True
        )
    axes = axes if n_rows > 1 else [axes]  # Handle single row case

    for i, hierarchy in enumerate(hierarchies):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j] if n_rows > 1 else axes[j]

            series = preprocess_target(targets_dict[dataset][hierarchy])

            sns.histplot(
                series, 
                bins=30, 
                color='royalblue', 
                alpha=0.7, 
                edgecolor='black', 
                ax=ax
                )

            if i == 0:  # Top row titles for datasets
                ax.set_title(f'{dataset}', fontsize=14)
            if j == 0:  # First column labels for hierarchies
                ax.set_ylabel(f'{hierarchy}\nFrequency', fontsize=12)
            ax.set_xlabel(EC_LABEL, fontsize=10)

            ax.set_xlim(x_min, x_max)

    fig.suptitle(
        r'Distribution of $\mathit{EC}$ Across Datasets and NAICS Hierarchies',
        size=16)

    plt.figtext(0.5, -0.02, NOTE, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()

    if write_path:
        plt.gcf().savefig(write_path)
#endregion

#region: chemical_coverage_heatmap
def chemical_coverage_heatmap(y, write_path=None):
    '''
    Creates a heatmap showing binary presence/absence (detections only) 
    of chemicals by sector.
    '''
    # Process data for presence/absence
    binary_presence = (y > 0).astype(int)
    # Count occurrences (1 = present, 0 = absent) for each pair
    coverage = (
        binary_presence
        .groupby(['DTXSID', 'naics_id'])
        .max()
        .unstack(fill_value=0)
    )

    chemical_totals = coverage.sum(axis=1).sort_values(ascending=False)
    sector_totals = coverage.sum(axis=0).sort_values(ascending=False)

    coverage = coverage.loc[chemical_totals.index, sector_totals.index]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        coverage,
        cmap='Blues',
        linewidths=0.5,
        linecolor='gray',
        cbar=False,   # No colorbar since it's binary
        annot=False   # Keep it clean without annotations
    )

    plt.title('Chemical Coverage by Sector (Presence/Absence)', fontsize=14)
    plt.xlabel('NAICS Sector')
    N_chemicals = len(y.index.get_level_values('DTXSID').unique())
    plt.ylabel(f'{N_chemicals} Chemicals (DTXSIDs)')
    # plt.xticks(rotation=45, ha='right')
    plt.yticks([])  # Omit y-axis ticklabels for 500+ chemicals
    plt.tight_layout()

    plt.figtext(
        0.5, -0.02,
        'Note: "Presence" reflects nonzero values; nondetects treated as "absent."',
        wrap=True, horizontalalignment='center', fontsize=10
    )

    if write_path:
        plt.gcf().savefig(write_path)
#endregion

#region: naics_level_data_summary
def naics_level_data_summary(y, moe_series, write_path=None):
    '''
    Generates scatterplot showing sector richness with a single legend for
    pre-aggregated concentration data.
    '''
    # Compute summary metrics for each sector
    summary = y.groupby(level='naics_id').agg(
        total_observations=('count'),
        nondetects=(lambda x: (x == 0).sum()),
        num_chemicals=(lambda x: x.index.get_level_values('DTXSID').nunique())
    )

    summary['proportion_nondetects'] = (
        summary['nondetects'] / summary['total_observations']
    )

    summary['median_log10(MOE)'] = moe_series.groupby('naics_id').median()

    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(
        data=summary,
        x='proportion_nondetects',
        y='num_chemicals',
        hue='median_log10(MOE)',
        palette='coolwarm_r',
        s=150,
        alpha=0.8,
        edgecolor='black',
        legend=False
    )

    # Add sector labels
    for i in summary.index:
        x = summary.loc[i, 'proportion_nondetects']
        y = summary.loc[i, 'num_chemicals']
        plt.annotate(
            f'Sec. {i}',  # Sector code
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            ha='center',
            fontsize=10,
            weight='bold'
        )

    plt.title('Data Quality & General Noncancer Risk By NAICS Sector', fontsize=16)
    plt.xlabel('Proportion of Non-Detects', fontsize=12)
    plt.ylabel('Number of Chemicals Represented', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    sm = plt.cm.ScalarMappable(cmap='coolwarm_r')
    sm.set_array(summary['median_log10(MOE)'])
    plt.colorbar(sm, label=r'$\log_{10}(\mathit{MOE})$')

    plt.tight_layout()

    if write_path:
        plt.gcf().savefig(write_path)
#endregion

#region: twa_concentrations_by_naics
def twa_concentrations_by_naics(y_by_sampling_no, write_path=None):
    '''
    Visualizes exposure distributions across sectors for multiple chemicals 
    using multi-panel boxplots.
    '''
    y_by_sampling_no = preprocess_target(y_by_sampling_no)

    chemicals = y_by_sampling_no.index.get_level_values('DTXSID').unique()

    n_chemicals = len(chemicals)
    n_rows = int(np.ceil(n_chemicals / 4))  # 4 columns per row
    fig, axes = plt.subplots(
        n_rows, 
        4, 
        figsize=(16, 4 * n_rows), 
        sharex=True, 
        sharey=True
        )
    axes = axes.flatten()  # Flatten for easy indexing

    # Generate plots for each chemical
    for i, chemical_id in enumerate(chemicals):
        # Filter data for the chemical
        data = y_by_sampling_no.xs(chemical_id, level='DTXSID').reset_index()

        sns.boxplot(
            ax=axes[i],
            data=data,
            x='naics_id',
            y='concentration',
            palette='coolwarm_r',
            showfliers=True
        )

        axes[i].set_title(f'{chemical_id}', fontsize=12)
        axes[i].set_xlabel('')

        # Show y-axis labels only in the first column
        if i % 4 == 0:  # First column (4 panels per row)
            axes[i].set_ylabel(EC_LABEL, fontsize=10)  # Show label
        else:
            axes[i].set_ylabel('')  # Remove label

        axes[i].tick_params(axis='x', rotation=45, labelsize=8)
        axes[i].grid(True, linestyle='--', alpha=0.6)

    # Hide any unused axes in the layout
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Main title and layout
    fig.suptitle('Distributions of Worker $\mathit{EC}$s by Sector for Data-Rich Chemicals', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])

    # Add caption (note) below the plot
    fig.text(0.5, 0.01, NOTE, ha='center', fontsize=10, style='italic')
    
    if write_path:
        fig.savefig(write_path)
#endregion

#region: correlation_by_naics
def correlation_by_naics(y, X, write_path=None):
    '''
    Creates multi-panel scatterplots with correlation coefficient for each 
    NAICS code.
    '''
    y = preprocess_target(y)
    X['logKoa'] = np.log10(X['KOA_pred'])
    merged = y.to_frame(name='log_concentration').join(X, how='inner')
    merged.reset_index(inplace=True)
    # Filter out any missing values for pearsonr
    merged = merged.dropna(subset=['logKoa', 'log_concentration'])

    sns.set(style='whitegrid', font_scale=1.2)
    unique_naics = sorted(merged['naics_id'].unique())
    n_codes = len(unique_naics)

    n_cols = 4
    n_rows = int(np.ceil(n_codes / n_cols))

    fig, axes = plt.subplots(
        n_rows, 
        n_cols, 
        figsize=(4 * n_cols, 4 * n_rows), 
        sharey=True
        )
    axes = axes.flatten() if n_rows > 1 else [axes]

    # Calculate global limits
    x_min = merged['logKoa'].min()
    x_max = merged['logKoa'].max()
    y_min = merged['log_concentration'].min()
    y_max = merged['log_concentration'].max()

    # Add padding (adjust as needed, e.g., 5% of the range)
    x_pad = 0.05 * (x_max - x_min)
    y_pad = 0.05 * (y_max - y_min)

    for i, naics in enumerate(unique_naics):
        ax = axes[i]
        subset = merged[merged['naics_id'] == naics]

        # Scatterplot with regression line
        sns.scatterplot(
            x='logKoa', 
            y='log_concentration', 
            data=subset, 
            ax=ax, 
            alpha=0.7
            )
        sns.regplot(
            x='logKoa', 
            y='log_concentration', 
            data=subset, 
            scatter=False, 
            color='red', 
            ci=None, 
            ax=ax
            )

        # Calculate Spearman's Rank Correlation
        corr, p_value = pearsonr(
            subset['logKoa'], 
            subset['log_concentration']
            )

        # Title with stats
        ax.set_title(f'NAICS: {naics}\n$r$={corr:.2f}, p={p_value:.3g}')
        ax.set_xlabel(r'$\log_{10}(K_{oa})$')
        ax.set_ylabel(r'$\log_{10}(\mathit{EC})$ (mg $\cdot$ m$^{-3}$)')

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        # Set even-numbered ticks
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(2))

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add suptitle and note
    fig.suptitle(
        'Correlation Between $\mathit{EC}$ and $K_{oa}$ By NAICS Sector', 
        fontsize=18,
        fontweight='bold')
    plt.figtext(0.5, 0.01, NOTE, ha='center', fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    if write_path:
        plt.savefig(write_path)
#endregion

#region: preprocess_target
def preprocess_target(y, log10=True):
    '''
    Handles non-detects and applies log-transformation to a target variable.
    '''
    # Replace zeros (non-detects) with half the smallest non-zero value
    non_zero_min = y[y > 0].min()
    small_value = 0.5 * non_zero_min
    y = y.replace(0, small_value)
    
    # Apply log transformation
    if log10:
        y = np.log10(y)
        
    return y
#endregion