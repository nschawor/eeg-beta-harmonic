all: processing figures

processing:
	python3 proc1_sensor_alpha_frequency.py
	python3 proc2_compute_ssd.py
	python3 proc3_spec_param_on_ssd.py

figures:
	python3 fig1a_rhythms_simulated.py
	python3 fig1b_rhythms_empirical.py
	python3 fig2a_percentile_spectrum_simulation.py
	python3 fig2b_C3_sensor_space.py
	python3 fig3a_alpha_examples.py
	python3 fig3b_harmonic_beta.py
