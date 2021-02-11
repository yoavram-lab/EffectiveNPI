# Inferring the effective start dates of non-pharmaceutical interventions during COVID-19 outbreaks
## Ilia Kohanovski, Uri Obolski, [Yoav Ram](http://www.yoavram.com)

Repository for paper:

> Kohanovski I, Obolski U, Ram Y. Inferring the effective start dates of non-pharmaceutical interventions during COVID-19 outbreaks. medRxiv. 2020. doi:[10.1101/2020.05.24.20092817](http://doi.org/10.1101/2020.05.24.20092817)

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

# License

Source code and results released under CC-BY-SA 4.0 license.
