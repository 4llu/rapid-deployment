# Fewshot framework

Aleksanteri Hämäläinen\
aleksanteri.hamalainen@aalto.fi\
Aalto University\
Mechatronics Research group

## Instructions

Run main code in `src` directory with

```
python main.py --config CONFIG_DIR/CONFIG_FILE
```

where `CONFIG_DIR` is a directory in `configs` and `CONFIG_FILE` to a `.yaml` config file in that directory.

To instead run hyperparameter optimization, use

```
python h_optimize.py --config CONFIG_DIR/CONFIG_FILE --n_trials NUM_TRIALS --n_repetitions NUM_REPETITIONS
```

where `NUM_TRIALS` is the total number of trials to run and `NUM_REPETITIONS` is the number of repetitions to run per trial (optional: default 1). Trial repetitions attempt to
make sure the found hyperparameters are stable.

## Notes

### Config


### Models

