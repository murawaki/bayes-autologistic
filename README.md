Bayesian autologistic model for linguistic typological features
===================
# About

  Yugo Murawaki and Kenji Yamauchi.
  [A Statistical Model for the Joint Inference of Vertical Stability and Horizontal Diffusibility of Typological Features](https://doi.org/10.1093/jole/lzx022).
  Journal of Language Evolution 3(1), pp. 13-25, 2018.

  An earlier, non-Bayesian version of the code can be found at https://github.com/yustoris/autologistic-coling-2016.

# Requirements

- Python3
    - numpy
    - networkx
    - matplotlib (for visualization)
    - [WALS data (language.csv)](http://wals.info/download) should be downloaded to `data/wals` directory

# How to Run the Model

## Missing value imputation

```
make -j -f eval.make mvi OUTDIR=`pwd`/data/standardmodel
```

### Parameter estimation

```
make -j -f eval.make param OUTDIR=`pwd`/data/standardmodel
```
