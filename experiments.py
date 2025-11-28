import time
import yaml
import argparse
import pathlib

import warnings
warnings.filterwarnings("ignore", message=".*no current CUDA context.*")

from utils import Buckley_Leverett, gen_data, train_model, evaluate_model


experiment_folder = pathlib.Path("./experiments/")
experiment_folder.mkdir(parents=True, exist_ok=True)
experiment_folder = experiment_folder.resolve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=int, default=0)
    args = parser.parse_args()


    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)


    # --- baseline ---
    baseline_params = params["baseline"]
    model_params = baseline_params["model"]
    times = params["times"]
    M = baseline_params["physics"]["M"]
    ic, bc, colloc, test = baseline_params["data"].values()
    hidden_layers, hidden_layer_neurons = model_params["hidden_layers"], model_params["hidden_layer_neurons"]
    adam_params = model_params["adam_params"]
    lbfgs_params = model_params["lbfgs_params"]
    X_ic, y_ic, X_bc, y_bc, X, X_tests = gen_data(ic, bc, colloc, test, times)

    if args.baseline:
        model = Buckley_Leverett(hidden_layers, hidden_layer_neurons)
        start = time.time()

        model.train()
        train_model(model, X, X_ic, y_ic, X_bc, y_bc, M, adam_params, lbfgs_params,
                    save=True, model_path="./baseline/model.pth",
                    save_history=True, history_path="./baseline/train_history.json")

        model.eval()
        evaluate_model(model, X_tests, times, M, save=True, path="./baseline/evaluation.json")
        print(f"Time elapsed: {((time.time() - start) / 60):.2f} min")


    # --- experiments ---
    experiments_params = params.get("experiments")

    if experiments_params is not None:
        experiment_number = 1
        start = time.time()
        for lr in experiments_params["lbfgs_params"]["lr"]:
            for max_iter in experiments_params["lbfgs_params"]["max_iter"]:
                for history_size in experiments_params["lbfgs_params"]["history_size"]:
                    print(f"Experiment #{experiment_number}")

                    lbfgs_params["lr"] = lr
                    lbfgs_params["max_iter"] = max_iter
                    lbfgs_params["history_size"] = history_size

                    file_postfix = f"{lr}_{max_iter}_{history_size}"

                    model = Buckley_Leverett(hidden_layers, hidden_layer_neurons)

                    model.train()
                    train_model(model, X, X_ic, y_ic, X_bc, y_bc, M,
                                adam_params, lbfgs_params,
                                save=True, model_path=f"{experiment_folder}/model_{file_postfix}.pth",
                                save_history=True, history_path=f"{experiment_folder}/train_history_{file_postfix}.json")

                    model.eval()
                    evaluate_model(model, X_tests, times, M,
                                save=True, path=f"{experiment_folder}/evaluation_{file_postfix}.json")

                    experiment_number += 1
        print(f"Overral experiment count is {experiment_number}\nTime elapsed: {((time.time() - start) / 60):.2f} min")
