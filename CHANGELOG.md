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


## [1.0.0] - 2026/XX/XX

### New Features
- **`data_io`**
  - Added function `save_json` which saves a `dict` into a JSON file.

### Enhanced
- **`maths`**
  - `rceil` and `rfloor` now return a scalar instead of `np.float`.

### Fixed
- **`core_ml`**
  - `percent_error` did not raise a ```ValueError``` if `true` was a list and contained zeros.
  - `r2_score_adj` would raise `TypeError``` if `n_predictors` was an `np.int`. It now validates the input with `numbers.Integral`.
  - `r2_score_adj` would allow cases where `n_predictors` was below zero. An error is raised is now raised in those cases.
  
### Maintenance
- **`package`**
  - Import name changed from `dscience_toolbox` to `dstools` to make it easier and faster to type in import statements.

- **`maths`**
  - New submodule `maths` created to store general purpose functions involving mathematical operations. #
  - The script `math_utils.py` was moved into this submodule and renamed into `arithmetic.py`.
  
- **`data_io`**
  - Script `data_misc.py` was renamed to `paths.py` for improved clarity of its content.
  - Path input in `list_files` was being unnecessarily coerced twice to `pathlib.PurePath`.
  

## [0.4.1] - 2026/06/17
Release 0.4.1 introduces improved set up of colours in plotting functions and more unit tests in the ```core_stats``` submodule.

### Enhanced
- **`visualisation`**
  - Added argument `colours` to all plotting functions to allow user to set colours directly instead of using argument ```plot_kwargs```.

### Maintenance
- **`core_stats`**
  - Added more unit tests to script ```test_inference.py``` and some of the unit tests no longer rely on random draws (which caused tests to occasionally fail due to sampling variability).
  - Unit tests in script ```test_effect_sizes.py``` no longer rely on random draws (which caused tests to occasionally fail due to sampling variability).


## [0.4.0] - 2026/06/12
Release 0.4.0 introduces a new histogram visualisation function and new utility functions for decimal ceiling and floor rounding. It also improves plotting 
argument validation, fixes issues in ```scatterplot``` and ```cohens_d```, reorganizes modules for better maintainability, and expands test coverage across 
descriptive statistics and diagnostics functionality.

### New Features
- **`visualisation`**
  - Added `histplot` for creation of histograms.
  
- **`utils`**
  - Added `rceil` to perform ceil rounding with decimal precision.
  - Added `rfloor` to perform floor rounding with decimal precision.

### Enhanced
- **`visualisation`**
  - Plotting functions now check if input arguments and ```plot_kwargs``` have duplicated keys, raising an warning. Function's input argument values take precedence over those in ```plot_kwargs```.

### Fixed
- **`visualisation`**
  - Fixed the ```vline``` argument in function ```scatterplot``` which was plotting an horizontal line instead of a vertical one.

- **`core_stats`**
  - ```cohens_d``` now returns zero instead of raising a ```ZeroDivisionError``` in edge cases where the denominator of the formulas were zero.

### Maintenance
- **`data_io`**
  - Create new submodule ```data_io``` to store functions that handle data and files. 

- **`core_stats`**
  - Moved effect size functions and data diagnostics functions (e.g. multicollinearity) to the scripts ```effect_sizes.py``` and ```diagnostics.py```, respectively.
  - Added more unit tests to script ```descriptive.py```.
  - Added more unit tests to script ```diagnostics.py```.
  

## [0.3.0] - 2026/06/05
Release 0.3.0 adds new machine learning and statistical utilities, including percentage error, adjusted R², and normality checks, 
along with a file listing helper. It also improves the stability of vif by automatically removing constant-value columns.

### New Features
- **`core_ml`**
  - Added `percent_error` to estimate % error element-wise between two arrays.	
  - Added `r2_score_adj` to estimate adjusted R2.	
  
- **`core_stats`**
  - Added `check_normality` to assess if a distribution is normal.	
	
- **`utils`**
  - Added `list_files` to list paths of files of one or more extensions.
  
### Fixed
- **`core_stats`**
  - Columns with constant values are now dropped in `vif` to avoid numerical unstability.

## [0.2.0] - 2026/05/03
Release 0.2.0 expands the library with new statistical tools (including permutation and rank-based tests), improved categorical
and descriptive summaries, and a new scatterplot function for enhanced visualisation. It also introduces basic I/O utilities 
for loading text and JSON files. A few plotting inconsistencies were fixed.

### New Features
- **`core_stats`**
  - Added `rank_based_tests` to perform non-parametric tests on two samples (Mann-Whitney and Wilcoxon).
  - Added `permutation_test_means` to perform a permutation test of the difference of means between two independent or paired samples.
  - Added `categorical_summary_stats` to calculate summary statistics of categorical variables.
  
- **`utils`**
  - Added `load_txt` to read .txt files as a list or DataFrame.
  - Added `load_json` to read json files.

- **`visualisation`**
  - Added `scatterplot` for creation of scatterplots.

### Enhanced
- **`core_stats`**
  - Empty columns in `summary_stats` are now explicitly removed to make the output cleaner.

### Fixed
- **`visualisation`**
  - The `axes_names` argument in `barplot` was not overwriting the labels of the x and y axes.
  - The font size of annotations in `barplot` was being incorrectly controlled by a deprecated parameter instead of `font_sizes` argument.
  - Default `seaborn` legend when `grp_var` is set in plotting functions is explicitly removed first before making the legend to avoid potential legend issues.
  
### Maintenance
- **`core_stats`**
  - Corrected returned output types in docstring of `t_test()`.


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