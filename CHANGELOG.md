# Changelog
All notable changes to this project will be documented in this file.


## Categories

- **New Features**  
  For new functions introduced to the package.

- **Enhanced**  
  For improvements to existing functionality (performance, usability, consistency, etc...).

- **Fixed**  
  For corrections of incorrect or unintended behaviour.

- **Maintenance (Internal Changes)**  
  For refactoring, dependency updates, code cleanup, documentation improvements, and other non-functional changes.

## [0.1.0] - 2026/04/26
Project initialisation. Contains three submodules: **`core_stats`**, **`visualisation`** and **`utils`**. 

### New Features
- **`core_stats`**
  - Added `summary_stats` for detailed descriptive statistics of a DataFrame.
  - Added `column_dtypes` to get dtypes of columns in a DataFrame.
  - Added `vif` for calculation of variance inflation factors.
  - Added `correlation` for calculation of pairwise correlation.
  - Added `cohens_d` for calculation of Cohen's d effect size (only for two-group independent and paired t-tests).
  - Added `t_test` to perform t-test (two-sample independent and paired).

- **`visualisation`**
  - Added `barplot()` for creation of barplots.
  - Added `boxplot()` for creation of boxplots. 

### Maintenance
- **`utils`**
  - Added `_to_numpy` to coerce input objects to a `numpy` array.
  - Added `_validate_pandas_dataframe` to check if an input is a `pandas` DataFrame.
  - Added `_check_plotting_inputs` to check if variables to be plotted exist in input DataFrame.  