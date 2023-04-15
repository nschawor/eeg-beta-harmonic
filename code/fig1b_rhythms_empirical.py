# %% FIG 1B
import matplotlib.pyplot as plt
import mne
from params import DATA_DIR, FIG_DIR, SPEC_NR_SECONDS
from helper import despine
import ssd
import matplotlib.ticker as mticker
import numpy as np

plt.style.use('figures.mplstyle')
plt.rc('axes.formatter', useoffset=False)

data = {'sub-032425': ('beta', 25, 'eo', 0, 47789),
        'sub-032311': ('alpha', 10, 'eo', 1, 17500),
        'sub-032339': ('alpha', 10, 'ec', 0, 46237),
        }
subjects = list(data.keys())

# %%
fig, ax = plt.subplots(3, 2,
                       gridspec_kw={'left': 0.05,
                                    'right': 0.85,
                                    'bottom': 0.15,
                                    'top': 0.77,
                                    'width_ratios': [2, 1]})


for i_sub, subject in enumerate(subjects):
    subject = subjects[i_sub]
    band, peak, condition, i_comp, idx = data[subject]

    file_name = f"{DATA_DIR}/{subject}_{condition}-raw.fif"
    raw = mne.io.read_raw_fif(file_name, preload=True)
    raw.pick_types(eeg=True)
    peak_beta = data[subject][1]
    SSD_WIDTH = 3

    signal_bp = [peak_beta - SSD_WIDTH, peak_beta + SSD_WIDTH]
    noise_bp = [peak_beta - (SSD_WIDTH + 2), peak_beta + (SSD_WIDTH + 2)]
    noise_bp = [None, peak_beta + (SSD_WIDTH + 2)]
    noise_bs = [peak_beta - (SSD_WIDTH + 1), peak_beta + (SSD_WIDTH + 1)]
    filters, patterns = ssd.compute_ssd(raw, signal_bp, noise_bp, noise_bs)

    raw_ssd = ssd.apply_filters(raw, filters[:, :2])

    n_fft = int(SPEC_NR_SECONDS * raw.info['sfreq'])
    psd, freq = mne.time_frequency.psd_welch(raw_ssd, fmin=1, fmax=45,
                                             n_fft=n_fft)

    raw_ssd.filter(1, None)

    length = 500
    signal = raw_ssd._data[i_comp][idx:idx + length]
    ax[i_sub, 0].plot(raw.times[:length], signal, 'k', lw=0.85)
    ax[i_sub, 0].set(xlim=(0.7, 1.3), xlabel='time [s]', yticks=[])
    ax[i_sub, 0].axis('off')

    ax[i_sub, 1].loglog(freq, psd[i_comp], lw=0.85,
                        color='k', zorder=-3 * i_sub)

    ax[i_sub, 1].xaxis.set_minor_formatter(mticker.NullFormatter())
    ax[i_sub, 1].set(xlim=(5, 40),
                     ylim=(np.min(psd), np.max(psd)),
                     xticks=[10, 20, 30],
                     xticklabels=[10, 20, 30],
                     yticks=[])
    despine(ax[i_sub, 1])

ax[2, 1].set(xlabel='frequency [Hz]')
ax[0, 1].set(ylabel='log power')

fig_name = f'{FIG_DIR}/fig1b_rhythms_empirical.pdf'
fig.set_size_inches(4, 4)
fig.savefig(fig_name, dpi=200, transparent=True)
fig.show()
