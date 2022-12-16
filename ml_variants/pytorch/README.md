### Setup

```bash
module load intel/20
```

```bash
module swap intel gcc
```

```bash
module load python/3.8.6
```

> Follow https://sites.google.com/a/case.edu/hpcc/hpc-cluster/software/software-installation-guide/anaconda-and-miniconda and activate your conda env

```bash
conda install --file requirements.txt
```

```bash
# train model and capture time taken
time python3 mlp.py
```
