# %% FIG 2B
import numpy as np
import matplotlib.pyplot as plt
import mne
from params import DATA_DIR, BETA_FMIN, BETA_FMAX, ALPHA_FMIN, \
    ALPHA_FMAX, FIG_WIDTH, FIG_DIR, SPEC_NR_SECONDS
from helper import percentile_spectrum, get_participant_list, despine
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
import scipy.stats
import matplotlib.ticker as mticker

plt.rc('axes.formatter', useoffset=False)
plt.style.use('figures.mplstyle')

mne.set_log_level(verbose=False)

# %% compute increase in alpha for each participant
condition = 'eo'
subjects = get_participant_list('data', condition)
corr = np.zeros((len(subjects),))
corr_p = np.zeros((len(subjects),))

nr_lines = 20
colors = plt.cm.viridis(np.linspace(0, 1, nr_lines + 1))

for i_sub, subject in enumerate(subjects):
    print(subject)
    file_name = f"{DATA_DIR}/{subject}_{condition}-raw.fif"

    raw = mne.io.read_raw_fif(file_name, preload=True)
    raw.set_eeg_reference('average')
    raw.pick_channels(['C3'])

    band = [BETA_FMIN, BETA_FMAX]
    psd_perc, freq = percentile_spectrum(raw, band=band,
                                         nr_lines=nr_lines,
                                         nr_seconds=SPEC_NR_SECONDS)

    beta1 = np.mean(psd_perc[:, (freq > band[0]) & (freq < band[1])], axis=1)

    aband = [ALPHA_FMIN, ALPHA_FMAX]
    idx_freq = (freq > aband[0]) & (freq < aband[1])
    alpha1 = np.mean(psd_perc[:, idx_freq], axis=1)

    corr[i_sub], corr_p[i_sub] = scipy.stats.spearmanr(alpha1, beta1)

# %% plot results and selected participants
subjects_sel = ['sub-032499', 'sub-032517', 'sub-032412', 'sub-032311']
colors2 = ['#FF2150', '#2751FF', '#FFC00D', '#19FF3F']

fig = plt.figure()
gs = gridspec.GridSpec(2, 2, left=0.1,
                       right=0.95, bottom=0.125,
                       top=0.9,
                       width_ratios=[1, 1.5],
                       height_ratios=[0.7, 1],
                       hspace=0.4)

# plot alpha-beta correlation
ax = plt.subplot(gs[0, :])
ax.hist(corr, 40, histtype='stepfilled', color='black')

despine(ax)

xmax = 31
corr_min = 0.45
subjects = list(subjects)
for i_sub, subject in enumerate(subjects_sel):
    idx = subjects.index(subject)
    ax.plot(corr[idx], xmax - 3, marker='s',
            markersize=5,
            color=colors2[i_sub])

# ax.plot([-1, -corr_min - 0.1], [xmax, xmax], color='k')
ax.text(-0.9, xmax + 2, 'case 1')
# ax.plot([-corr_min, corr_min], [xmax, xmax], color='darkgrey')
ax.text(0, xmax + 2, 'case 2')
# ax.plot([corr_min + 0.1, 1], [xmax, xmax], color='k')
ax.text(0.7, xmax + 2, 'case 3')

ax.set(xlabel='Spearman correlation beta-power with alpha-power',
       ylabel='participants',
       xticks=np.arange(-1, 1.01, 0.5))

# % plot empirical spectra
gs2 = gridspec.GridSpecFromSubplotSpec(1, 4, gs[1, :])

for i_sub, subject in enumerate(subjects_sel):

    ax = plt.subplot(gs2[i_sub])
    file_name = f"{DATA_DIR}/{subject}_{condition}-raw.fif"
    raw = mne.io.read_raw_fif(file_name, preload=True)
    raw.pick_types(eeg=True)
    raw.set_eeg_reference('average')

    band = [BETA_FMIN, BETA_FMAX]

    # plot percentile spectra
    nr_lines = 5
    colors = plt.cm.viridis(np.linspace(0, 1, nr_lines + 1))

    leg = ['top-20%', ' ', ' ', '', 'lowest 20%']
    idx = mne.pick_channels(raw.ch_names, ['C3'])
    i_chan = idx[0]
    psd_perc, freq = percentile_spectrum(raw, band=band,
                                         nr_seconds=SPEC_NR_SECONDS,
                                         nr_lines=nr_lines,
                                         i_chan=i_chan)
    for i in range(psd_perc.shape[0]):
        ax.loglog(freq, psd_perc[i], lw=0.85,
                  color=colors[i], zorder=-3 * i, label=leg[i])
    ax.axvspan(band[0], band[1], color='gray', alpha=0.2, zorder=-10)

    ax.set(xlim=(5, 45), xticks=[10, 20, 30],
           xlabel='frequency [Hz]', yticks=[])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='x')

    ax.spines['bottom'].set_color(colors2[i_sub])
    ax.spines['top'].set_color(colors2[i_sub])
    ax.spines['right'].set_color(colors2[i_sub])
    ax.spines['left'].set_color(colors2[i_sub])

    # plot spatial patterns
    filter = np.zeros(len(raw.ch_names))
    filter[idx] = 1
    raw.filter(BETA_FMIN, BETA_FMAX)
    cov = np.cov(raw._data)
    pattern = filter @ cov
    ax_ins = inset_axes(ax, width='40%', height='40%', loc='lower left')
    mne.viz.plot_topomap(pattern, raw.info, axes=ax_ins, show=False)

    if i_sub == 0:
        ax.set(ylabel='log power [a. u.]')


fig.set_size_inches(FIG_WIDTH, 4)
fig_name = f'{FIG_DIR}/fig2b_C3_sensor_space.pdf'
fig.savefig(fig_name, dpi=200, transparent=True)
fig.show()
