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

## Data

### ARotor

-

### ARotor replication


#### Missing

**Fixed**
* 1000rpm_CT_baseline_11%_GP8_0 data missing in Jesses conversions (0 bytes)
* 500rpm_CT_baseline_6%_GP6_0 is missing from original files (_motor data does exist though)

**Not fixed**
* 1000rpm_CT_failure_11%_GP2_0.csv missing (dataset 2)
* 1000rpm_CT_failure_11%_GP5_0.csv missing (dataset 2)

* DS2 1000rpm 11%
** Missing `severe_tff` and `severe_wear`

* All baseline 250rpm
** Maybe use one of the new healthy ones and simulate gearpairs with different installations?

#### Comments

Either leave be or move to manual changes if fixed.

**Weirdly high**

* GP1_mild_pitting/data_set_1/3012Hz/sensor/1500rpm_CT_failure_1%_GP1_0.feather
** torque2 higher than rest for mild pitting (mean 10 VS 5). Other sensors seem normal
** Action: leave be

**Encoder direction**

Encoder 4 runs in the wron direction in most files (too many to list here).

**Encoders weirdly low**

All encoders scaled very low (4 * 1e-7) compared to everywhere else

* GP1_mild_pitting/data_set_2/3012Hz/sensor/500rpm_CT_failure_6%_GP1_0.feather
* GP1_mild_pitting/data_set_2/3012Hz/sensor/750rpm_CT_failure_11%_GP1_0.feather
* GP1_mild_pitting/data_set_2/3012Hz/sensor/1000rpm_CT_failure_6%_GP1_0.feather

#### anual changes

**Recording window off**

* GP7_mild_wear/data_set_2/3012Hz/sensor/1250rpm_CT_failure_11%_GP7_0.feather
** Cut 15 000 samples from start (had slow start)

* GP9_mild_tff/data_set_2/3012Hz/sensor/250rpm_CT_failure_11%_GP9_0.feather
** Cut 180 000 samples from end (recording window seems to be a bit off)

**Swapped sensors**

* GP4_mild_micropitting/data_set_2/3012Hz/sensor/1500rpm_CT_failure_6%_GP4_0.feather
** Swapped Torq1 and Torq2

* GP5_mild_micropitting/data_set_2/3012Hz/sensor/1500rpm_CT_failure_11%_GP4_0.feather
** Swapped Torq1 and Torq2

* GP5_severe_tff/data_set_2/3012Hz/sensor/1500rpm_CT_failure_6%_GP5_0.feather
** Swapped Torq1 and Torq2

* GP5_severe_tff/data_set_2/3012Hz/sensor/1500rpm_CT_failure_11%_GP5_0.feather
** Swapped Torq1 and Torq2
