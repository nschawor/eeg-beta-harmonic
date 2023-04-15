# %%
import mne
from params import SPEC_PARAM_DIR, CSV_DIR, SSD_DIR, SPEC_NR_PEAKS, \
    SPEC_NR_SECONDS, SSD_PARAM_DIR, ALPHA_FMIN, ALPHA_FMAX, SNR_THRESHOLD

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fooof
from helper import percentile_spectrum, get_participant_list

os.makedirs(SPEC_PARAM_DIR, exist_ok=True)
subjects = pd.read_csv(f"{CSV_DIR}/name_match.csv")
subjects = subjects.INDI_ID
conditions = ("eo", "ec")
condition = 'eo'

os.makedirs(SSD_PARAM_DIR, exist_ok=True)


def process_1sub(subject, condition):
    spec_file = f"{SSD_PARAM_DIR}/{subject}_{condition}.csv"

    ssd_fname = f'{SSD_DIR}/{subject}_{condition}_raw.fif'
    raw_ssd = mne.io.read_raw_fif(ssd_fname)

    band = (ALPHA_FMIN, ALPHA_FMAX)
    psd_perc, freq = percentile_spectrum(raw_ssd,
                                         band=band,
                                         i_chan=0,
                                         nr_lines=4,
                                         nr_seconds=SPEC_NR_SECONDS)

    fm = fooof.FOOOF(max_n_peaks=SPEC_NR_PEAKS)
    fm.fit(freqs=freq, power_spectrum=psd_perc[0])

    alpha = fooof.analysis.get_band_peak_fm(fm, [ALPHA_FMIN, ALPHA_FMAX])
    amplitude = alpha[1]
    if amplitude < SNR_THRESHOLD:
        return

    psd_corr = fm.power_spectrum - fm._ap_fit

    # find alpha frequency
    idx_start = np.argmin(np.abs(freq - band[0]))
    idx_end = np.argmin(np.abs(freq - band[1]))

    idx_max = np.argmax(psd_corr[idx_start:idx_end]) + idx_start
    peak_freq = freq[idx_max]

    # find beta frequency
    idx_start = np.argmin(np.abs(freq - 2 * band[0]))
    idx_end = np.argmin(np.abs(freq - 2 * band[1]))

    idx_max = np.argmax(psd_corr[idx_start:idx_end]) + idx_start
    beta_freq = freq[idx_max]

    # create dataframe with data
    df_subject = pd.Series(
        data={"subject": subject,
              "alpha_peak": peak_freq,
              'beta_peak': beta_freq}
    )
    df_subject.to_csv(spec_file)
    plt.close('all')


# %% compute for all participants
for condition in ['eo', 'ec']:
    subjects = get_participant_list('ssd', condition)
    for subject in subjects:
        # process_1sub(subject, condition)
        print(subject)

    # compile all subject specific files
    subjects = get_participant_list('ssd_param', condition)
    print(len(subjects))

    dfs = []
    for i_sub, subject in enumerate(subjects):
        spec_file = f"{SSD_PARAM_DIR}/{subject}_{condition}.csv"
        df = pd.read_csv(spec_file, index_col=0)
        dfs.append(df)

    df_all = pd.concat(dfs, axis=1).T
    df_all.columns = ('subject', 'alpha_peak', 'beta_peak')
    df_all.to_csv(f'{CSV_DIR}/ssd_param_{condition}.csv', index=False)
