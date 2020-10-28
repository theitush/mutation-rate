from matplotlib import pyplot as plt


def set_plots_size_params(size):
    # Adapted from https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
    bigger = size * 1.2
    slightly_bigger = size * 1.1
    smaller = size * 0.8
    plt.rc('font', size=size)                        # controls default text sizes
    plt.rc('axes', titlesize=bigger)                 # fontsize of the axes title
    plt.rc('axes', labelsize=slightly_bigger)        # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size)                  # fontsize of the tick labels
    plt.rc('ytick', labelsize=size)                  # fontsize of the tick labels
    plt.rc('legend', fontsize=smaller)                  # legend fontsize


def get_mutation_data(freqs):
    mutation_data = freqs[(
        ((freqs.Freq>0.001) & (freqs.Freq<0.5)) | (freqs.Pos!=round(freqs.Pos))) &
        (freqs.Read_count>50) &
        (freqs.Pos>=23) & (freqs.Pos<=3549)].copy()
    mutation_data = mutation_data.sort_values(by=['passage'])
    coverages = freqs[['Read_count', 'Pos', 'passage', 'km']].sort_values(
        by=['km', 'passage', 'Pos']).set_index(['km', 'passage', 'Pos'])
    mutation_data['coverage'] = mutation_data.apply(
        lambda row: coverages.loc[(row.km, row.passage, round(row.Pos))].Read_count[0], axis=1)
    mutation_data.loc[mutation_data.Ref=="-", "Freq"] = mutation_data.loc[mutation_data.Ref=="-", "Read_count"] * \
                                                        mutation_data.loc[mutation_data.Ref=="-", "Freq"] / \
                                                        mutation_data.loc[mutation_data.Ref=="-", "coverage"]
    return mutation_data[(mutation_data.Freq>0.001) & (mutation_data.Freq<0.5)]


def mutation_time_series(df, ax, colors_dict, going=None):
    x = df.passage
    y = df.Freq
    if going=='up':
        if y.iloc[0] >= y.iloc[-1]*0.7:
            return
    if going=='down':
        if y.iloc[0] <= y.iloc[-1]*0.7:
            return
    label = f"{df.Pos.iloc[0]}{df.Base.iloc[0]}"
    if label in colors_dict.keys():
        color = colors_dict[label]
        alpha = 1
    else:
        color = 'black'
        label = None
        alpha = 0.1
    ax.plot(df.passage, df.Freq, color=color, label=label, alpha=alpha)
    # ax.text(x.iloc[-1], y.iloc[-1], label)  # <- text next to plot!
    return ax


def muts_overview(data, ax):
    plt.figure(figsize=(20, 10))
    base_color = {'A': 'red',
                  'G': 'blue',
                  'C': 'orange',
                  'T': 'green',
                  '-': 'black'}
    for letter in data.Base.unique():
        letter_df = data[data.Base==letter]
        for phase in letter_df.passage.unique():
            phase_df = letter_df[letter_df.passage==phase]
            y = phase_df[f'Freq']
            ax.scatter(phase_df.Pos, y, label=f"{letter}{phase}", color=base_color[letter], alpha=1.1-(int(phase)/10))
    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_ylim([-0.02,0.53])
    ax.set_xlabel('Postion')
    ax.set_ylabel('Frequency')
    return ax


def plot_overview(freqs):
    fig, axes = plt.subplots(figsize=(60, 30), ncols=3, nrows=3)
    plt.suptitle("MS2 Time Series AccuNGS MOI 0.1 with Read_count>50", fontsize=18)
    plt.subplots_adjust(wspace=0.3, hspace=0.25)
    print("Getting Mutation Data...")
    mutation_data = get_mutation_data(freqs)
    top_mutations = mutation_data.groupby(['Pos','Base']).Freq.max().sort_values(ascending=False).head(20).index
    top_mutations = [f'{pos}{base}' for (pos, base) in top_mutations]
    color_list = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
                  '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
                  '#000075', '#000000']
    color_dict = {top_mutations[i]: color_list[i] for i in range(len(top_mutations))}
    print("Creating graphs!")
    for i in range(3):
        km_muts = mutation_data[mutation_data.km==i+1]
        axes[0][i] = muts_overview(km_muts[km_muts.Freq>0.01], axes[0][i])
        axes[0][i].set_title(f"Mutations Overview Sample {i+1}")
        km_muts.groupby(['Pos', 'Base']).apply(
            lambda df: mutation_time_series(df, axes[1][i], color_dict, going='up'))
        axes[1][i].set_ylim([-0.01,0.5])
        axes[1][i].legend(bbox_to_anchor=(1, 0.7))
        axes[1][i].set_title(f"Mutations Increasing in Frequency in Sample {i+1}")
        axes[1][i].set_xlabel('Passage')
        axes[1][i].set_ylabel('Frequency')
        km_muts.groupby(['Pos', 'Base']).apply(
            lambda df: mutation_time_series(df, axes[2][i], color_dict, going='down'))
        axes[2][i].set_ylim([-0.01,0.4])
        axes[2][i].set_xlabel('Passage')
        axes[2][i].set_ylabel('Frequency')
        axes[2][i].set_title(f"Mutations Decreasing in Frequency in Sample {i+1}")
        axes[2][i].legend(bbox_to_anchor=(1, 1))
    return fig, axes