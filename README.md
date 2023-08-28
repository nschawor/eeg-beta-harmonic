# Overcoming harmonic hurdles: genuine beta-band rhythms vs. contributions of alpha-band waveform shape

This repository provides analysis code to visualize beta-activity in a large open EEG dataset. 

## Reference

Schaworonkow, N.: Overcoming harmonic hurdles: genuine beta-band rhythms vs. contributions of alpha-band waveform shape. Imaging Neuroscience (2023). Retrieved from ![https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00018/](direct.mit.edu/imag/article/doi/10.1162/imag_a_00018)

## Dataset

The results are based on following available openly available data set: ["Leipzig Cohort for Mind-Body-Emotion Interactions" (LEMON dataset)](http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html), from which we used the preprocessed EEG data. The associated data set research article: 
- Babayan A et al.: [A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults.](http://www.nature.com/articles/sdata2018308) _Scientific Data_ (2018).

## Requirements

The provided python3 scripts are using ```scipy``` and ```numpy``` for general computation, ```pandas``` for saving intermediate results to csv-files. ```matplotlib``` for visualization. For EEG-related analysis, the ```mne``` package is used. For computation of aperiodic exponents: [```specparam```](https://specparam-tools.github.io/). 

# Pipeline
To reproduce the figures from the command line, navigate into the ```code``` folder and execute ```make all```. This will run through the preprocessing steps and generate the figures. The scripts can also be executed separately in the order described in the ```Makefile```. If data is not converted into fif-format yet, the ```proc0_convert_data_to_mne.py```-script should be executed.
