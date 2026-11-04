# MECH

## Sensate project workflow / 项目评估入口

Run the integrated baseline from the project root:

```bash
python evaluate/run_mech_routing.py --experiment all
python evaluate/plot_result_figures.py
```

Baseline data is written to `evaluate/result/baseline_mech/<experiment>/`;
project figures are written to `evaluate/figure/`. See
[Evaluation Workflow](../../evaluate/evaluate.md) for supported experiment axes
and model constraints.

The notebook instructions below describe the standalone MECH experiments and
their table/figure numbering, not the Sensate evaluation outputs. Run these
notebooks from `baseline/MECH/` so their relative data paths resolve correctly.

## Standalone notebook dependencies
* **python >= 3.9.6**
* **numpy == 1.23.5**
* **matplotlib == 3.7.2**
* **qiskit == 0.45.1**
* **networkx == 3.1**
* **jupyter notebook**


## How to Run
#### Open the following jupyter notebook files to reproduce experimental data of baseline and MECH for corresponding table and figures.
* Qiskit_experiments.ipynb (Table 2, Fig. 12,13,14,15,16)
    * You may select `optimization_level=2` instead of `optimization_level=3` for a faster run; validate the resulting metrics
* MECH_experiments.ipynb (Table 2, Fig. 12,13,14,16)
* MECH_experiments_customization.ipynb (Fig. 15)

#### The experimental data will be automatically saved to `./baseline_data` and `./exp_data`. 
* The paths can be changed at the beginning of the jupyter notebooks above
* Sample baseline data can be found in `./sample_baseline_data`
* Sample MECH data can be found in `./sample_exp_data`

#### Then, open the following jupyter notebook files to read from the saved data and generate the table and plots automatically.
* MECH_main_table.ipynb (Table 2)
* MECH_line_plot.ipynb (Fig. 12, 13)
* MECH_bar_plot.ipynb (Fig. 14, 15, 16)

## What to Run
#### 1. Experimental setup for baseline

Open Qiskit_experiments.ipynb
* `iteration = 5` or customized
* `mode = 'level_3' or 'level_2' or 'sabre'`

Table 2
* `structure = 'square', chiplet_array_dim = (3,3), chiplet_size = (6,6), sparsity = None`
* `structure = 'square', chiplet_array_dim = (3,3), chiplet_size = (7,7), sparsity = None`
* `structure = 'square', chiplet_array_dim = (3,3), chiplet_size = (8,8), sparsity = None`
* `structure = 'square', chiplet_array_dim = (3,3), chiplet_size = (9,9), iterations = 5, sparsity = None`

Fig. 12
* `structure = 'square', chiplet_array_dim = (2,2), chiplet_size = (7,7), sparsity = None`
* `structure = 'square', chiplet_array_dim = (2,3), chiplet_size = (7,7), sparsity = None`
* `structure = 'square', chiplet_array_dim = (3,3), chiplet_size = (7,7), sparsity = None`
* `structure = 'square', chiplet_array_dim = (3,4), chiplet_size = (7,7), sparsity = None`

Fig. 13
* `structure = 'square', chiplet_array_dim = (3,3), chiplet_size = (7,7), sparsity = None`

Fig.14
* `structure = 'square', chiplet_array_dim = (3,3), chiplet_size = (7,7), sparsity = None`
* `structure = 'square', chiplet_array_dim = (3,3), chiplet_size = (7,7), sparsity = 3`
* `structure = 'square', chiplet_array_dim = (3,3), chiplet_size = (7,7), sparsity = 1`

Fig. 15
* `structure = 'square', chiplet_array_dim = (2,3), chiplet_size = (9,9), sparsity = None`
* `structure = 'square', chiplet_array_dim = (2,3), chiplet_size = (9,9), data_qubit_num = 366, sparsity = None`
* `structure = 'square', chiplet_array_dim = (2,3), chiplet_size = (9,9), data_qubit_num = 288, sparsity = None`

Fig. 16
* `structure = 'square', chiplet_array_dim = (3,3), chiplet_size = (7,7), iterations = 5, sparsity = None`
* `structure = 'hexagon', chiplet_array_dim = (2,3), chiplet_size = (8,8), iterations = 5, sparsity = None`
* `structure = 'hevay_square', chiplet_array_dim = (3,3), chiplet_size = (8,8), iterations = 5, sparsity = None`
* `structure = 'heavy_hexagon', chiplet_array_dim = (3,4), chiplet_size = (8,8), iterations = 5, sparsity = None`


#### 2. Experiemental setup for MECH
Open MECH_experiments.ipynb

Table 2
* `structure = 'square', chiplet_array_dim = (3,3), chiplet_size = (6,6), sparsity = None`
* `structure = 'square', chiplet_array_dim = (3,3), chiplet_size = (7,7), sparsity = None`
* `structure = 'square', chiplet_array_dim = (3,3), chiplet_size = (8,8), sparsity = None`
* `structure = 'square', chiplet_array_dim = (3,3), chiplet_size = (9,9), sparsity = None`

Fig. 12
* `structure = 'square', chiplet_array_dim = (2,2), chiplet_size = (7,7), sparsity = None`
* `structure = 'square', chiplet_array_dim = (2,3), chiplet_size = (7,7), sparsity = None`
* `structure = 'square', chiplet_array_dim = (3,3), chiplet_size = (7,7), sparsity = None`
* `structure = 'square', chiplet_array_dim = (3,4), chiplet_size = (7,7), sparsity = None`

Fig. 13
* `structure = 'square', chiplet_array_dim = (3,3), chiplet_size = (7,7), sparsity = None`

Fig.14
* `structure = 'square', chiplet_array_dim = (3,3), chiplet_size = (7,7), sparsity = None or 3 or 1` (results similar)

Fig. 15
* `structure = 'square', chiplet_array_dim = (2,3), chiplet_size = (9,9), sparsity = None`
* Open MECH_experiments_customization.ipynb and run all

Fig. 16
* `structure = 'square', chiplet_array_dim = (3,3), chiplet_size = (7,7), sparsity = None`
* `structure = 'hexagon', chiplet_array_dim = (2,3), chiplet_size = (8,8), sparsity = None`
* `structure = 'hevay_square', chiplet_array_dim = (3,3), chiplet_size = (8,8), sparsity = None`
* `structure = 'heavy_hexagon', chiplet_array_dim = (3,4), chiplet_size = (8,8), sparsity = None`

#### 3. Plotting

Open MECH_main_table.ipynb (Table 2)
* `baseline_mode = 'level_3' or 'level_2' or 'sabre'` matching the selected experiment configuration
* `baseline_data_path = 'baseline_data'`
* `exp_data_path = 'exp_data'`

Open MECH_line_plot.ipynb (Fig. 12, 13)
* `baseline_mode = 'level_3' or 'level_2' or 'sabre'` matching the selected experiment configuration
* `baseline_data_path = 'baseline_data'`
* `exp_data_path = 'exp_data'`

Open MECH_bar_plot.ipynb (Fig. 14, 15, 16)
* `baseline_mode = 'level_3' or 'level_2' or 'sabre'` matching the selected experiment configuration
* `baseline_data_path = 'baseline_data'`
* `exp_data_path = 'exp_data'`

