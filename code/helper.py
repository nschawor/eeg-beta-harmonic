import os
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt

from params import SSD_DIR, CSV_DIR, SPEC_NR_SECONDS, \
    DATA_DIR, SPEC_PARAM_DIR, SSD_PARAM_DIR


def percentile_spectrum(raw, band=(8, 12), nr_lines=5, i_chan=0,
                        nr_seconds=SPEC_NR_SECONDS):
    """ Function to compute the percentile spectrum: Cut the given signal into
    segments of [nr_seconds] length. Compute PSD for each segment. Sort the
    segments according to the power in a given frequency band. Divide into
    percentile groups and compute the average PSD for each group.

    Parameters
    ----------
        band [2 x 1]: frequency band for sorting the PSDs
        nr_lines (int): number of groups
        i_chan (int): channel index
        nr_seconds: segment length

    Returns
    -------
        psd_perc (array): percentile spectrum
        freq (array): frequency axis of computed spectrum

    """

    events = mne.make_fixed_length_events(raw,
                                          start=0,
                                          stop=raw.times[-1],
                                          duration=nr_seconds)

    epo = mne.Epochs(raw, events, tmin=0, tmax=nr_seconds, baseline=None)

    n_fft = nr_seconds * int(raw.info['sfreq'])
    psd, freq = mne.time_frequency.psd_welch(epo,
                                             picks=[i_chan],
                                             fmin=1,
                                             fmax=45,
                                             n_fft=n_fft)

    idx_start = np.argmin(np.abs(freq - band[0]))
    idx_end = np.argmin(np.abs(freq - band[1]))
    mean_power = np.mean(psd[:, 0, idx_start:idx_end], axis=1)
    idx_segments = np.argsort(mean_power, axis=0)[::-1]

    spacing = int(np.floor(len(events)/nr_lines))

    # compute percentiles
    psd_perc = np.zeros((nr_lines, psd.shape[-1]))
    for i in range(nr_lines):
        idx1 = idx_segments[i * spacing:(i + 1) * spacing]
        psd_perc[i] = np.mean(psd[idx1, 0, :], axis=0)

    return psd_perc, freq


def get_participant_list(aspect, condition):

    df = pd.read_csv(f'{CSV_DIR}/name_match.csv')
    subjects = df.INDI_ID
    subjects_selected = None
    if aspect == 'data':
        data_exists = np.zeros((len(subjects),), dtype='bool')
        for i_sub, subject in enumerate(subjects):
            fif_fname = f'{DATA_DIR}/{subject}_{condition}-raw.fif'
            data_exists[i_sub] = os.path.exists(fif_fname)
        subjects_selected = subjects[np.where(data_exists)[0]].to_list()

    elif aspect == 'ssd':
        ssd_exists = np.zeros((len(subjects),), dtype='bool')
        for i_sub, subject in enumerate(subjects):
            ssd_fname = f"{SSD_DIR}/{subject}_ssd_filters_{condition}.csv"
            ssd_exists[i_sub] = os.path.exists(ssd_fname)
        subjects_selected = subjects[np.where(ssd_exists)[0]].to_list()

    elif aspect == 'sensor_param':
        sparam_exists = np.zeros((len(subjects),), dtype='bool')
        for i_sub, subject in enumerate(subjects):
            sparam_fname = f"{SPEC_PARAM_DIR}/{subject}_{condition}.csv"
            sparam_exists[i_sub] = os.path.exists(sparam_fname)
        subjects_selected = subjects[np.where(sparam_exists)[0]].to_list()

    elif aspect == 'ssd_param':
        ssd_param_exists = np.zeros((len(subjects),), dtype='bool')
        for i_sub, subject in enumerate(subjects):
            ssd_param_fname = f"{SSD_PARAM_DIR}/{subject}_{condition}.csv"
            ssd_param_exists[i_sub] = os.path.exists(ssd_param_fname)
        subjects_selected = subjects[np.where(ssd_param_exists)[0]].to_list()

    subjects = np.sort(subjects_selected)

    return list(subjects)


# plot patterns
def plot_patterns(patterns, raw, nr_components, colors, cmap="RdBu_r"):
    """Plot a number of spatial patterns as a topography.

    Parameters
    ----------
        patterns (array): spatial patterns to plot.
        raw (mne.io.Raw): raw-file for electrode positions.
        nr_components (int): Number of components that will be plotted.
        colors (list): List of identifying colors.
        gs (matplotlib.gridspec.GridSpec): grid for plotting the topographies.
        cmap (str, optional): Colormap for topographies. Defaults to "RdBu_r".
    """

    fig, ax = plt.subplots(2, 5)
    dd = 0
    cc = 0
    for i in range(nr_components):
        ax1 = ax[dd, cc]

        idx1 = np.argmax(np.abs(patterns[:, i]))
        patterns[:, i] = np.sign(patterns[idx1, i]) * patterns[:, i]
        mne.viz.plot_topomap(patterns[:, i], raw.info, axes=ax1,
                             cmap=cmap, show=False)
        ax1.set_title("      ", backgroundcolor=colors[i], fontsize=8)

        cc += 1
        if cc == 5:
            dd += 1
            cc = 0
    fig.set_size_inches(8, 3)

    return fig


def despine(ax):
    if type(ax) == np.ndarray:
        for axx in ax.flat:
            axx.spines['right'].set_visible(False)
            axx.spines['top'].set_visible(False)
    else:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
