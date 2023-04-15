# parameters and paths
DATA_DIR = "/Users/n_schaworonkow/data/eeg/eeg-lemon/"
BASE_DIR = "/Users/n_schaworonkow/projects/eeg-alpha-beta/"

CSV_DIR = f"{BASE_DIR}/csv/"
RESULTS_DIR = f"{BASE_DIR}/results/"

# spectral parametrization
SPEC_PARAM_DIR = f"{RESULTS_DIR}/sensor_param/"
SSD_PARAM_DIR = f"{RESULTS_DIR}/ssd_param/"
SSD_DIR = f'{RESULTS_DIR}/ssd/'

ALPHA_FMIN = 8
ALPHA_FMAX = 13

BETA_FMIN = 16
BETA_FMAX = 30

FRAC_DEVIATION = 0.05

SPEC_FMIN = 2
SPEC_FMAX = 35
SPEC_NR_PEAKS = 5
SPEC_NR_SECONDS = 3
SSD_WIDTH = 2
SNR_THRESHOLD = 0.5

FIG_DIR = f"{BASE_DIR}/figures/"
FIG_WIDTH = 8
