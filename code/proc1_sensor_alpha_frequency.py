# %%
""" Data: compute center frequency for each EEG subject."""
import os
import mne
import numpy as np
import pandas as pd
import fooof

from params import DATA_DIR, \
    SPEC_FMIN, SPEC_FMAX, \
    SPEC_NR_SECONDS, SPEC_PARAM_DIR, \
    SPEC_NR_PEAKS, ALPHA_FMAX, ALPHA_FMIN
from helper import get_participant_list


def process_1sub(subject, condition):
    os.makedirs(SPEC_PARAM_DIR, exist_ok=True)
    spec_file = f"{SPEC_PARAM_DIR}/{subject}_{condition}.csv"

    file_name = f"{DATA_DIR}/{subject}_{condition}-raw.fif"
    raw = mne.io.read_raw_fif(file_name, preload=True)
    raw.set_eeg_reference("average")

    if subject == "sub-032478":
        # this participants has potentially wrong-labeled channel names
        return

    # pick midline channels
    raw.pick_types(eeg=True)
    midline_channels = [ch for ch in raw.ch_names if "z" in ch]
    raw.pick_channels(midline_channels)
    front_channels = [ch for ch in raw.ch_names if "F" in ch]
    raw.drop_channels(front_channels)

    # compute PSD
    psd, freqs = mne.time_frequency.psd_welch(
        raw,
        fmin=SPEC_FMIN,
        fmax=SPEC_FMAX,
        n_fft=int(SPEC_NR_SECONDS * raw.info["sfreq"]),
        n_overlap=raw.info["sfreq"],
    )

    # fit spec param
    fm = fooof.FOOOFGroup(max_n_peaks=SPEC_NR_PEAKS)
    fm.fit(freqs, psd)
    alpha_bands = fooof.analysis.get_band_peak_fg(fm, [ALPHA_FMIN, ALPHA_FMAX])

    peak = np.nanmean(alpha_bands[:, 0])
    amp = np.nanmean(alpha_bands[:, 1])
    rsq = np.mean([fm.get_results()[i][2] for i in range(len(raw.ch_names))])

    # create dataframe with data
    df_subject = pd.Series(
        data={"subject": subject,
              "peak_frequency": peak,
              "peak_amplitude": amp,
              'rsq': rsq}
    )
    df_subject.to_csv(spec_file)

    return peak


# %%
if __name__ == "__main__":

    for condition in ['eo', 'ec']:
        subjects = get_participant_list('data', condition)
        for subject in subjects:
            peak = process_1sub(subject, condition)
            print(subject, peak)
