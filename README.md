<h1 align="center">SUEGE</h1>

## Installation

The repo requires python>=3.7, anaconda and a new env is recommended.

``` sh
conda create -n surge python=3.7 -y # optional
conda activate surge # optional
pip install -e .
```

### Data

First download the standard benchmark datasets. The Data folder can be downloaded from [GDELT & ICEWS14/05-15](https://github.com/BorealisAI/de-simple/tree/master/datasets), [ICEWS18](https://github.com/TemporalKGTeam/xERTE/tree/main/tKGR/data/ICEWS18_forecasting), [YAGO11k & WikiData12k](https://drive.google.com/open?id=1S0dcMDXVZp8CFSCMojkBQI1gCva8Dm-0). Then process the dataset using the commands below.

```sh
cd data
# for GDELT/ICEWS14/ICEWS05-15/ICEWS18
# e.g. python preprocess.py icews14
python preprocess.py $dataset_name
# for YAGO11k and WikiData12k
python preprocess_interval.py $dataset_name
```

### Training

Configurations for the experiments are in the `/config` folder.

``` sh
python -m kge start config/surge/gdelt-best.yaml
```

The training process uses DataParallel in all visible GPUs by default, which can be overrode by appending `--job.device cpu` to the command above.

### Evaluation

You can evaluate the trained models on dev/test set using the following commands.

``` sh
python -m kge eval config/surge/gdelt-best.yaml --checkpoint<saved_dir>
python -m kge test config/surge/gdelt-best.yaml --checkpoint <saved_dir>/checkpoint_best.pt
```

## Acknowledgment

Thanks [LibKGE](https://github.com/uma-pi1/kge) and [HittER](https://github.com/microsoft/HittER) for providing the preprocessing scripts and the base frameworks.

