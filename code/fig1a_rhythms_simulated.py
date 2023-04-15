# %% FIG 1A
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import neurodsp.sim
from helper import despine
from params import FIG_DIR, SPEC_NR_SECONDS
import matplotlib.ticker as mticker

plt.style.use('figures.mplstyle')
plt.rc('axes.formatter', useoffset=False)
os.makedirs(FIG_DIR, exist_ok=True)

# %% settings
np.random.seed(22)
nr_seconds = 120
nr_samples = 1000
time = np.linspace(0, nr_seconds, nr_seconds * nr_samples)
freq = 10
alpha = np.sin(2 * np.pi * freq * time)
beta = np.sin(2 * np.pi * 2 * freq * (time - 0.015))

# %% simulate same powerlaw signals
powerlaw1 = neurodsp.sim.sim_powerlaw(n_seconds=nr_seconds,
                                      fs=nr_samples, exponent=-2)
offset = 1

# %% simulate 3 different conditions
x = np.zeros((3, len(alpha)))

# cond 1: negative
alpha_amp = np.ones_like(alpha)
x[0] = (powerlaw1 * (alpha_amp * alpha)) + offset

# cond 2: independent
beta_amp = 1
x[1] = (powerlaw1 * (beta_amp * 0.25 * beta)) + offset

# cond 3: harmonic
alpha_amp = np.ones_like(alpha)
beta_amp = np.ones_like(alpha)
x[2] = powerlaw1 * (alpha_amp * alpha + beta_amp * 0.25 * beta) + offset

ch_names = ['eeg1', 'eeg2', 'eeg3']
info = mne.create_info(ch_names, sfreq=nr_samples, ch_types='eeg')
raw = mne.io.RawArray(x, info)
n_fft = SPEC_NR_SECONDS * nr_samples
psd, freq = mne.time_frequency.psd_welch(raw, fmax=45, n_fft=n_fft)
raw.filter(1, 45)

# %% plot signals + power spectra
fig, ax = plt.subplots(3, 2,
                       gridspec_kw={'left': 0.05,
                                    'right': 0.85,
                                    'bottom': 0.15,
                                    'top': 0.77,
                                    'width_ratios': [2, 1]})

for i_chan in range(3):
    idx_start = 1200
    idx_end = idx_start + 600
    ax[i_chan, 0].plot(time[idx_start:idx_end],
                       raw._data[i_chan][idx_start:idx_end],
                       lw=0.85, color='k')
    ax[i_chan, 0].set(yticks=[], xticks=[0])
    ax[i_chan, 0].axis('off')
    ax[i_chan, 1].loglog(freq, psd[i_chan], lw=0.85,
                         color='k', zorder=-3 * i_chan)

    ax[i_chan, 1].xaxis.set_minor_formatter(mticker.NullFormatter())
    ax[i_chan, 1].set(xlim=(5, 40),
                      ylim=(np.min(psd), np.max(psd)),
                      xticks=[10, 20, 30],
                      xticklabels=[10, 20, 30],
                      yticks=[])

ax[2, 1].set(xlabel='frequency [Hz]')
ax[0, 1].set(ylabel='log power')
despine(ax)
fig.set_size_inches(4, 4)
fig_name = f'{FIG_DIR}/fig1a_rhythms_simulated.pdf'
fig.savefig(fig_name, dpi=200, transparent=True)
