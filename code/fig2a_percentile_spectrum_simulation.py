# %% FIG 2A
import numpy as np
import matplotlib.pyplot as plt
import mne
import neurodsp.sim
from helper import percentile_spectrum, despine
from params import BETA_FMIN, BETA_FMAX, FIG_WIDTH, FIG_DIR
import matplotlib.ticker as mticker

plt.style.use('figures.mplstyle')
plt.rc('axes.formatter', useoffset=False)

# %% settings
np.random.seed(22)
nr_seconds = 120
nr_samples = 1000
time = np.linspace(0, nr_seconds, nr_seconds * nr_samples)
freq = 10
alpha = np.sin(2 * np.pi * freq * time)
beta = np.sin(2 * np.pi * 2 * freq * time)

# %% simulate same powerlaw signals
powerlaw1 = neurodsp.sim.sim_powerlaw(n_seconds=nr_seconds,
                                      fs=nr_samples, exponent=-1.5)
powerlaw2 = neurodsp.sim.sim_powerlaw(n_seconds=nr_seconds,
                                      fs=nr_samples, exponent=-1.5)
offset = .5 * neurodsp.sim.sim_powerlaw(n_seconds=nr_seconds,
                                        fs=nr_samples, exponent=-1)

# %% simulate 3 different conditions
x = np.zeros((3, len(alpha)))

# cond 1: negative
alpha_amp = np.ones_like(alpha)
alpha_amp[int(len(time)/2):] = 0

beta_amp = 0 * np.ones_like(alpha)
beta_amp[int(len(time)/2-10000):] = 1
x[0] = (powerlaw1 * (alpha_amp * alpha) + (beta_amp * 0.25 * beta) * powerlaw2) + offset

# cond 2: independent
alpha_amp = 1 + np.random.randn(len(alpha))
beta_amp = 1 + np.random.randn(len(alpha))

x[1] = (powerlaw1 * (alpha_amp * alpha) + (beta_amp * 0.25 * beta) * powerlaw2) + offset

# cond 3: harmonic
alpha_amp = np.ones_like(alpha)
beta_amp = np.ones_like(alpha)
x[2] = powerlaw1 * (alpha_amp * alpha + beta_amp * 0.25 * beta) + offset


info = mne.create_info(['eeg1', 'eeg2', 'eeg3'], sfreq=nr_samples, ch_types='eeg')
raw = mne.io.RawArray(x, info)

# %% plot power spectra
nr_lines = 5
colors = plt.cm.viridis(np.linspace(0, 1, nr_lines + 1))
band = [BETA_FMIN, BETA_FMAX]
labels = [r'$\bf{case\ 1}$' + '\n beta is anticorrelated\n to alpha',
          r'$\bf{case\ 2}$' + '\n beta & alpha are \nindependent',
          r'$\bf{case\ 3}$' + '\n beta is a harmonic\n of alpha']
leg = ['top 20%\nsegments', ' ', ' ', ' ', 'lowest 20%\nsegments']

fig, ax = plt.subplots(1, 3, gridspec_kw={'left': 0.05,
                       'right': 0.85, 'bottom': 0.15, 'top': 0.77})
for i_chan in range(3):
    psd_perc, freq = percentile_spectrum(raw, band=band, nr_seconds=2,
                                         nr_lines=nr_lines,
                                         i_chan=i_chan)
    for i in range(psd_perc.shape[0]):
        ax[i_chan].loglog(freq, psd_perc[i], lw=0.85,
                          color=colors[i], zorder=-3 * i, label=leg[i])
    ax[i_chan].axvspan(band[0], band[1], color='gray', alpha=0.2, zorder=-10)
    ax[i_chan].xaxis.set_minor_formatter(mticker.NullFormatter())
    ax[i_chan].set(xlim=(5, 40),
                   xlabel='frequency [Hz]',
                   xticks=[10, 20, 30],
                   xticklabels=[10, 20, 30],
                   title=labels[i_chan],
                   yticks=[])
    # if i_chan == 0:
    #     ax[0].text(12, 1.5*np.max(psd_perc), , va='top')
ax[0].set(ylabel='log power [a. u.]')
title = f'percentile spectrum:\nsegments sorted by\nbeta-power ({band[0]}–{band[1]} Hz)'
ax[2].legend(title=title, bbox_to_anchor=[0.825, 0.32],
             bbox_transform=fig.transFigure)
despine(ax)

fig.set_size_inches(FIG_WIDTH, 3)
fig.tight_layout()
fig_name = f'{FIG_DIR}/fig2a_percentile_spec_simulation.pdf'
fig.savefig(fig_name, dpi=200, transparent=True)


# %%
