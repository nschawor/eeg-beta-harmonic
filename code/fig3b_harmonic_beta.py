# %% FIG 3B
import matplotlib.pyplot as plt
from params import FIG_DIR, FIG_WIDTH, CSV_DIR, FRAC_DEVIATION
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
from helper import despine

plt.style.use('figures.mplstyle')

# %% aggregate over all participants
fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [2, 1, 2]})

colors = ('#18A12A', '#A31B40')
labels = {'ec': 'eyes closed', 'eo': 'eyes open'}
labels_n = {'ec': 'eyes\nclosed', 'eo': 'eyes\nopen'}

np.random.seed(22)

for i_cond, condition in enumerate(['ec', 'eo']):
    ax_group = ax[2 * i_cond]
    df = pd.read_csv(f'{CSV_DIR}/ssd_param_{condition}.csv')
    print(pg.pairwise_corr(df))

    df['fraction'] = df['beta_peak'] / df['alpha_peak']

    # jitter just for visualization
    df['alpha_peak'] += 0.05*np.random.randn(len(df))
    df['beta_peak'] += 0.05*np.random.randn(len(df))

    # allowed deviation for determining color
    df['close'] = (df.fraction < 2 + FRAC_DEVIATION) & \
        (df.fraction > 2 - FRAC_DEVIATION)

    sns.scatterplot(x='alpha_peak',
                    y='beta_peak',
                    size=.08, data=df,
                    palette=colors,
                    alpha=0.75,
                    hue='close', ax=ax_group)

    alpha_range = np.linspace(7, 14, 100)
    ax_group.plot(alpha_range, 2 * alpha_range, 'gray', zorder=-5)
    ax_group.legend([], [], frameon=False)

    ax_group.set(xlim=(7, 13),
                 xlabel='alpha-frequency [Hz]',
                 ylim=(15, 26),
                 ylabel='beta-frequency [Hz]',
                 title=labels[condition])

    ax_bar = ax[1]
    prop = len(df[df['close']]) / len(df)
    ax_bar.bar([i_cond], prop, color=colors[1])
    ax_bar.bar([i_cond], bottom=prop, height=1 - prop, color=colors[0])
    x = i_cond - .35
    ax_bar.text(x, prop / 2, f'{100*prop:.1f}%', color='w')
    ax_bar.text(x, (prop + 1) / 2, f'{100*(1-prop):.1f}%', color='w')

    ax_bar.text(x, 1.025, f'N={len(df)}')

ax_bar.set(xticks=[0, 1],
           xticklabels=[labels_n['ec'],
           labels_n['eo']],
           yticks=[],
           ylabel='proportion of participants')

despine(ax)
fig.show()
fig.set_size_inches(FIG_WIDTH, 3)
fig.tight_layout()
fig_name = f'{FIG_DIR}/fig3b_alpha_frequencies.pdf'
fig.savefig(fig_name, transparent=True, dpi=200)
