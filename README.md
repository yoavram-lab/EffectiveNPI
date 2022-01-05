# Inferring the effective start dates of non-pharmaceutical interventions during COVID-19 outbreaks
## Ilia Kohanovski, [Uri Obolski](http://uriobols.wixsite.com), [Yoav Ram](http://www.yoavram.com)

Repository for paper:

> Kohanovski Ilia, Obolski Uri, Ram Yoav (2022) [Inferring the effective start dates of non-pharmaceutical interventions during COVID-19 outbreaks](https://doi.org/10.1016/j.ijid.2021.12.364). International Journal of Infectious Diseases. doi:10.1016/j.ijid.2021.12.364; also on [medRxiv](http://doi.org/10.1101/2020.05.24.20092817).

# Data

The data for 11 European countries is taken from [Imperial College COVID-19 Response Team](https://github.com/ImperialCollegeLondon/covid19model), Imperial College London, originally used in
> Flaxman S, Mishra S, Gandy A, Unwin HJT, Mellan TA, Coupland H, Whittaker C, Zhu H, Berah T, Eaton JW, Monod M, Ghani AC, Donnelly CA, Riley SM, Vollmer MAC, Ferguson NM, Okell LC, Bhatt S. Estimating the effects of non-pharmaceutical interventions on COVID-19 in Europe. Nature. 2020;(March):1-35. doi:[10.1038/s41586-020-2405-7](http://doi.org/10.1038/s41586-020-2405-7)

The data for Wuhan, China, was retrieved from [Shaman group, Columbia University](https://github.com/SenPei-CU/COVID-19), originally used in
> Li R, Pei S, Chen B, Song Y, Zhang T, Yang W, Shaman J. Substantial undocumented infection facilitates the rapid dissemination of novel coronavirus (SARS-CoV2). Science (80- ). March 2020:eabb3221. doi:[10.1126/science.abb3221](https://science.sciencemag.org/content/368/6490/489)

We duplicate the data in our repository in case the file is moved/updated.

# Abstract

During Feb-Apr 2020, many countries implemented non-pharmaceutical interventions, such as school closures and lockdowns, with variable schedules, to control the COVID-19 pandemic caused by the SARS-CoV-2 virus.
Overall, these interventions seem to have successfully reduced the spread of the pandemic.
We hypothesise that the official and effective start date of such interventions can significantly differ, for example due to slow adoption by the population, or because the authorities and the public are unprepared.
We fit an SEIR model to case data from~12 countries to infer the effective start dates of interventions and contrast them with the official dates.
We find mostly late, but also early effects of interventions. For example, Italy implemented a nationwide lockdown on Mar 11, but we infer the effective date on Mar 17 (+-2.99 days 95% CI). In contrast, Germany announced a lockdown on Mar 22, but we infer an effective start date on Mar 19 (+-1.05 days 95% CI).
We demonstrate that differences between the official and effective start of NPIs can distort conclusions about their impact, and discuss potential causes and consequences of our results.

# Inference results

Inference results (prior samples, reports) are saved to iCloud in the following folders:
- 2020-06-23-Mar28: inference up to Mar 28, 2020; 1M iterations; used to calibrate model
- 7M: inference up to Apr 11, 2020; 7M iterations; main model
- 7MFixed; inference up to Apr 11 with τ=τ* fixed at official data
- 7MFixedNoTau: inference up to Apr 11 without τ, ie no change point
- 2020-05-27-Apr4: inference up to Apr 4; 1M iterations
- 2020-05-25-Apr24: inference up to Apr 24; 1M iterations

# Instructions

To run the inference, run 
```
python src/inference.py
```
Run with the `-h` option for usage instructions

The script uses the case data from `data` folder and persists the inferred chains to `output-tmp/dir_name/inference/country_name.npz`. `dir_name` is defined by the date of the execution and the provided '--ver-desc' option (short description of the version). `dir_name` is then passed as an argument for scripts described below.

To reproduce all the figures, the following scripts are executed in order:
1. `make_report.ipynb` - set the `dir_name` variable and execute all cells
- analyzes inferred chains and persists summary table and plots to `dir_name/tables`, and `dir_name/figures`
2. `python Fig_tau_summary.py dir_name`
- uses summary report from the previous step and constructs `Fig-tau-summary.pdf` (Figures 1 and S5)
3. `python Fig_tau_posterior.py dir_name country_name/all -q`
- prepares `country_τ_posterior.pdf` figures (Figure 2)
- `tau.sh` can be used to run it for all countries
4. `python Fig_ppc.py dir_name country green/red/blue` - green/red/blue is the color of the plotted lines
- prepares `country_ppc_long.pdf` for Figure S4
- `ppc.sh` can be used to run it for all countries
5. `python Table_estimated_params.py dir_name`
- prepares Table 2
6. `python Fig_trace_tau.py dir_name country_name`
- prepares `country_trace.pdf` for Figure S3
7. `python Fig-autocorr.py dir_name`
- prepares `Fig-autocorr.pdf` for Figure S2
8. `compare_posteriors.ipynb`
- checks that different inference runs result in similar posterior
9. `python Fig_joint_tau_lambda.py dir_name country_name`
- produces `country_joint.pdf` for Figure S6.
- Notice, this has hardcoded number of burning steps (2M) and it removes one bad chain for Spain
10. `python Re.py dir_name country_name`
- prepare a `Re.csv` file that is necessary for executing `Re.ipynb` (Figure S7 Fig_RE2.pdf) and `Fig_Re.py` (Figure 4)
- `Re.sh` can be used to run for every country
11. `Re.ipynb` - change if needed the dir names (job_id_free, job_id_fixed) and run all cells
- prepares `Fig_RE2.pdf` for Figure S7
12. `python Fig_Re.py`
- prepares `Fig_RE.pdf` for Figure 4
13. `Table-WAIC.py`
- prepares Table-WAIC.csv by comparing different models (Table S2)
14. `Table-RMSE.py`
- prepares `Table-RMSE.csv` (Table S1)
15. `Fig_NPI_dates.py`
- prepares Figure S1

Other files:
- `model` folder contains all the models:
 - `NormalPriorModel` is the main model and other models inherit from it.
 - `NormalPriorModel` with truncanted normal distribution prior for τ. 
 - `UniformPriorModel` with uniform distribution prior for τ.
 - `FixedTauModel` with fixed τ defined by the official NPI date.
 - `NoTauModel` assumes there were no transmission and reporting rate changes for all the period.
 - `NormalPriorFreepModel` has additional parameters `Td1` and `Td2` with uniform prior that correspond to confirmation time in days before and after τ, respectively.
 - `NormalPriorNegativeBinModel` uses Negative Binomial distribution instead of Poisson for drawing the number of daily confirmed cases.
- `plot_utils.py` is the main file that is used for loading inferred chains and preparing plots. See `make_report.ipynb` for it usage in Jupyter
- `model_selection_reports` folder contains some additional model comparisons that were done

# License

- Source code: MIT License.
- Data: see original repositories.
- Manuscript: see publisher.
- Other content: <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
