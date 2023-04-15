# %%
""" Data: compute SSD filters for all subjects and save them.
"""
import pandas as pd
import mne
import numpy as np
import os
import ssd
from helper import get_participant_list
from params import DATA_DIR, SSD_DIR, SPEC_PARAM_DIR, SSD_WIDTH, \
    SNR_THRESHOLD

# %% specify participants and folders
os.makedirs(SSD_DIR, exist_ok=True)


def process_1sub(subject, condition):

    ssd_filters_fname = f"{SSD_DIR}/{subject}_ssd_filters_{condition}.csv"
    raw_ssd_file = f'{SSD_DIR}/{subject}_{condition}_raw.fif'

    spec_file = f"{SPEC_PARAM_DIR}/{subject}_{condition}.csv"
    df = pd.read_csv(spec_file, index_col=0)
    peak_alpha = df.T['peak_frequency'].to_numpy('float32')[0]
    peak_amp = df.T['peak_amplitude'].to_numpy('float32')[0]

    # load data
    file_name = f"{DATA_DIR}/{subject}_{condition}-raw.fif"
    raw = mne.io.read_raw_fif(file_name)
    raw.load_data()
    raw.pick_types(eeg=True)
    raw.set_montage('standard_1020')

    if np.isnan(peak_alpha):
        return
    if peak_amp < SNR_THRESHOLD:
        return

    print(f'running SSD with peak: {peak_alpha} Hz')
    filters, patterns = ssd.run_ssd(raw, peak=peak_alpha,
                                    band_width=SSD_WIDTH)

    df_filters = pd.DataFrame(filters.T, columns=raw.ch_names)
    df_filters.to_csv(ssd_filters_fname, index=False)

    df_patterns = pd.DataFrame(patterns.T, columns=raw.ch_names)
    ssd_patterns_fname = f"{SSD_DIR}/{subject}_ssd_patterns_{condition}.csv"
    df_patterns.to_csv(ssd_patterns_fname, index=False)

    nr_components = 4
    raw_ssd = ssd.apply_filters(raw, filters[:, :nr_components])
    raw_ssd.save(raw_ssd_file, overwrite=True)

    return


if __name__ == "__main__":

    for condition in ['eo', 'ec']:
        subjects = get_participant_list('sensor_param', condition)
        for subject in subjects:
            process_1sub(subject, condition)
            print(subject)
