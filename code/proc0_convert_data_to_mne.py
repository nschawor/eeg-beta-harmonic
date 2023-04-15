# %%
import os
import mne
import pandas as pd
import numpy as np

from params import DATA_DIR, CSV_DIR

data_DIR = DATA_DIR
new_data_DIR = '/cs/department2/data/eeg_lemon/raw_renamed/'
os.makedirs(new_data_DIR, exist_ok=True)


df = pd.read_csv(f'{CSV_DIR}/name_match2.csv')
subjects = df.INDI_ID

folder = '/cs/department2/data/eeg_lemon/epo_from_raw/'
os.makedirs(folder, exist_ok=True)

for i_sub, subject in enumerate(subjects):
    print(i_sub, subject)

    initial_name, bids_name = df.iloc[i_sub]

    os.makedirs(f"{data_DIR}/{subject}/RSEEG", exist_ok=True) 
    file_type = 'vhdr'
    new_file = f"{new_data_DIR}/{initial_name}/RSEEG/{initial_name}.{file_type}"

    # S200 eyes open, S10 eyes closed
    cond_list = {210: 'ec', 200: 'eo'}
    trigger = 210

    # load raw
    raw_file_name = f'{folder}/{subject}_{cond_list[trigger]}-raw.fif'
    if os.path.exists(raw_file_name):
        continue

    if not(os.path.exists(new_file)):
        continue

    raw = mne.io.read_raw_brainvision(new_file, eog=['VEOG'])
    raw.load_data()
    raw.filter(0.5, None)
    events, event_id = mne.events_from_annotations(raw)

    for trigger in cond_list.keys():

        raw_file_name = f'{folder}/{subject}_{cond_list[trigger]}-raw.fif'
        if os.path.exists(raw_file_name):
            continue

        condA = mne.pick_events(events, trigger)
        idx, = np.where(np.diff(condA[:, 0]) > 15000)
        condB = np.vstack((condA[0], condA[idx + 1]))
        epo = mne.Epochs(raw, condB, tmin=0, tmax=60, baseline=None)
        epo.load_data()

        data = np.vstack(np.transpose(epo._data, [0, 2, 1])).T
        raw2 = mne.io.RawArray(data, raw.info)

        raw2.save(raw_file_name)
