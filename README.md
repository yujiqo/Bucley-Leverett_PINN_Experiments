## Main project files description:
`utils.py` contains:
- analytical solution
- torch model
- data generation function
- training function
- evaluation function
- ploting function

`params.yaml` is a place where you specify hyperparams to experiment with.

`experiments.py` is an example how to use these functions to train multiple models with different hyperparams. <b>Note:</b> if you run experiments.py with --baseline flag it will train baseline model as well.

`baseline/` folder contains baseline model and data for plotting train history and prediction accuracy

<br>

## How to analyze data that we got after experiments?
i create ipynb file where i plot the training/evaluation data and from that point i can decide what to improve. Sample `analysis.ipynb` code:
```python
import yaml

from ..utils import plot_flux, plot_data
from ..experiments import experiment_folder


with open("../params.yaml", "r") as f:
    params = yaml.safe_load(f)
    M = params["baseline"]["physics"]["M"]
    experiments_params = params["experiments"]


plot_flux(M)
plot_data("Baseline Model Training", "Baseline Model Predictions",
          "./baseline/train_history.json", "./baseline/evaluation.json")


for lr in experiments_params["lbfgs_params"]["lr"]:
    for max_iter in experiments_params["lbfgs_params"]["max_iter"]:
        for history_size in experiments_params["lbfgs_params"]["history_size"]:
            file_postfix = f"{lr}_{max_iter}_{history_size}"

            plot_data(f"Model Training (L-BFGS lr={lr}, mi={max_iter}, hs={history_size})",
                    f"Model Predictions (L-BFGS lr={lr}, mi={max_iter}, hs={history_size})",
                    f"{experiment_folder}/train_history_{file_postfix}.json",
                    f"{experiment_folder}/evaluation_{file_postfix}.json")
```
