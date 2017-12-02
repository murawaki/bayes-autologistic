Bayesian autologistic model for linguistic typological features
===================
# About

  (TODO: add the new paper here)

  An earlier, non-Bayesian version of the code can be found at https://github.com/yustoris/autologistic-coling-2016

# Requirements

- Python3
    - numpy
    - networkx
    - matplotlib (for visualization)
    - WALS data (language.csv)](http://wals.info/download) should be downloaded to `data/wals` directory

# How to Run the Model

## Missing value imputation

```
make -j -f eval.make mvi OUTDIR=`pwd`/data/standardmodel
```

### Parameter estimation

```
make -j -f eval.make param OUTDIR=`pwd`/data/standardmodel
```
