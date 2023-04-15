# %% FIG 3A
import matplotlib.pyplot as plt
import mne
from params import DATA_DIR, FIG_WIDTH, FIG_DIR, SPEC_NR_SECONDS, \
    SSD_PARAM_DIR
import numpy as np
import ssd
import pandas as pd
from helper import percentile_spectrum, despine
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.style.use('figures.mplstyle')

# %% plot several example participants
nr_lines = 5
colors = plt.cm.viridis(np.linspace(0, 1, nr_lines + 1))

fig, ax = plt.subplots(1, 3)
subject_list = [('sub-032362', 'ec'),
                ('sub-032317', 'eo'),
                ('sub-032327', 'eo')]

for i_sub, (subject, condition) in enumerate(subject_list):

    df = pd.read_csv(f'{SSD_PARAM_DIR}/{subject}_{condition}.csv', index_col=0)
    # df = df.set_index('subject')

    peak = df.T['alpha_peak'].to_numpy('float')
    file_name = f"{DATA_DIR}/{subject}_{condition}-raw.fif"
    raw = mne.io.read_raw_fif(file_name, preload=True)
    raw.pick_types(eeg=True)

    filters, patterns = ssd.run_ssd(raw, peak=peak, band_width=2)
    raw_ssd = ssd.apply_filters(raw, filters[:, :2])

    band = [peak - 2, peak + 2]
    psd_perc, freq = percentile_spectrum(raw_ssd, band=band,
                                         nr_seconds=SPEC_NR_SECONDS,
                                         nr_lines=nr_lines)

    # plot percentile spectrum
    axB = ax.flat[i_sub]
    for i in range(psd_perc.shape[0]):
        axB.plot(freq, 10 * np.log10(psd_perc[i]),
                 color=colors[i], lw=0.85,
                 zorder=-3 * i)

    axB.axvspan(peak - 2, peak + 2, color='gray', alpha=0.15, zorder=-300)
    axB.axvline(peak, color='darkgrey', zorder=-100, lw=1, alpha=0.3)
    axB.axvline(2 * peak, color='darkgrey', zorder=-100, lw=1, alpha=0.3)

    if condition == 'eo':
        title = 'eyes open'
    else:
        title = 'eyes closed'

    axB.set(xlim=(0, 35),
            xticks=[0, 10, 20, 30],
            title=title,
            yticks=[])

    ax_ins = inset_axes(axB, width='40%', height='40%', loc=1)

    idx = np.argmax(np.abs(patterns[:, 0]))
    sign = np.sign(patterns[idx, 0])
    mne.viz.plot_topomap(sign * patterns[:, 0],
                         raw.info,
                         axes=ax_ins,
                         show=False)

for i in range(3):
    ax[i].set_xlabel('frequency [Hz]')
ax[0].set(ylabel='log power [a. u.]')
ax[0].text(0.5, -42.5, 'sorted \naccording\nto alpha-\npower')

despine(ax)
fig.set_size_inches(FIG_WIDTH, 3)
fig.tight_layout()
fig_name = f'{FIG_DIR}/fig3a_alpha_examples.pdf'
fig.savefig(fig_name, transparent=True, dpi=200)

# %%
