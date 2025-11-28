import torch
import json
import time
import numpy as np
import matplotlib.pyplot as plt


SEED = 2017

if torch.cuda.is_available():
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(SEED)
else:
    torch.manual_seed(SEED)

np.random.seed(SEED)


def f(S, M):
    return S ** 2 / (S ** 2 + (1 - S) ** 2 / M)

def df(S, M, h=1e-6):
    return (f(S + h, M) - f(S - h, M)) / (2 * h)

def shock_speed(S_L, S_R, M):
    if abs(S_L - S_R) < 1e-12:
        return df(S_L, M)

    return (f(S_L, M) - f(S_R, M)) / (S_L - S_R)

def solution(x, t, M, S_L=1.0, S_R=0.0, tol=1e-6):
    xi = x / t
    u = np.empty_like(xi)
    s = shock_speed(S_L, S_R, M)

    S_star = None
    for S in [i / 1000 for i in range(1, 1000)]:
        l = df(S, M)
        r = shock_speed(S, S_R, M)

        if abs(l - r) < tol:
            S_star = S
            break

    for i, xi_i in enumerate(xi):
        if S_star is None: # pure shock
            if xi_i < s:
                u[i] = S_L
            else:
                u[i] = S_R
        else:
            fp_S_L = df(S_L, M)
            fp_S_star = df(S_star, M)

            if xi_i < fp_S_L: # before fan
                u[i] = S_L
            elif xi_i < fp_S_star: # fan
                S_vals = np.linspace(0, 1, 2001)
                dS = np.abs(df(S_vals, M) - xi_i)
                u[i] = S_vals[np.argmin(dS)]
            elif xi_i < s: # shock
                u[i] = S_star
            else:
                u[i] = S_R # after shock

    return u


def pde(u, f, X, M, e=2.5e-3):
    du = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dx = du[:, 0:1]
    du_dt = du[:, 1:2]
    df_dx = torch.autograd.grad(f, X, grad_outputs=torch.ones_like(f), create_graph=True)[0][:, 0:1]
    d2u_dx2 = torch.autograd.grad(du_dx, X, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0][:, 0:1]

    return du_dt + df(u, M) * du_dx - e * d2u_dx2

class Buckley_Leverett(torch.nn.Module):
    def __init__(self, hidden_count, hidden_neurons, activation=torch.nn.Tanh()):
        super().__init__()

        layers = []
        layers.append(torch.nn.Linear(2, hidden_neurons))
        layers.append(activation)
        for _ in range(hidden_count):
            layers.append(torch.nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(activation)
        layers.append(torch.nn.Linear(hidden_neurons, 1))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)


def gen_data(ic, bc, colloc, test, times):
    x_ic = torch.rand(ic, 1)
    t_ic = torch.zeros_like(x_ic)
    X_ic = torch.cat([x_ic, t_ic], dim=1)
    y_ic = torch.zeros_like(x_ic)

    t_bc = torch.rand(bc, 1) + 1e-6
    x_bc = torch.zeros_like(t_bc)
    X_bc = torch.cat([x_bc, t_bc], dim=1)
    y_bc = torch.ones_like(x_bc)

    X = torch.rand(colloc, 2)
    X.requires_grad_(True)

    x_test = torch.linspace(0, 1, test).unsqueeze(1)
    X_tests = []
    for time in times:
        t_test = torch.full_like(x_test, time)
        X_test = torch.cat([x_test, t_test], dim=1)
        X_tests.append(X_test)

    return (X_ic, y_ic, X_bc, y_bc, X, X_tests)

def train_model(model, X, X_ic, y_ic, X_bc, y_bc, M,
                adam_params, lbfgs_params,
                save=False, model_path="./model_weights.pth",
                save_history=True, history_path="./history.json"):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    history = {"adam": [], "lbfgs": [], "time": {}}

    def loss_function(model, X, X_ic, y_ic, X_bc, y_bc, M):
        y_pred = model(X)

        residual = pde(y_pred, f(y_pred, M), X, M)
        loss_pde = torch.mean(residual ** 2)

        y_pred_ic = model(X_ic)
        loss_ic = torch.mean((y_pred_ic - y_ic) ** 2)

        y_pred_bc = model(X_bc)
        loss_bc = torch.mean((y_pred_bc - y_bc) ** 2)

        loss = loss_ic + loss_bc + loss_pde
        residual = torch.mean(residual)

        return (loss_ic, loss_bc, loss_pde, loss, residual)

    start = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_params["lr"])
    for _ in range(0 if bool(adam_params["use"]) else adam_params["epochs"] + 1, adam_params["epochs"] + 1):
        optimizer.zero_grad()
        loss_ic, loss_bc, loss_pde, loss, r = loss_function(model, X, X_ic, y_ic, X_bc, y_bc, M)
        loss.backward()
        optimizer.step()

        history["adam"].append({
            "ic": loss_ic.item(),
            "bc": loss_bc.item(),
            "pde": loss_pde.item(),
            "general": loss.item(),
            "residual": r.item()
        })
    end = (time.time() - start) / 60

    history["time"]["adam"] = end

    optimizer = torch.optim.LBFGS(model.parameters(),
                                  lr=lbfgs_params["lr"], max_iter=lbfgs_params["max_iter"],
                                  history_size=lbfgs_params["history_size"], line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss_ic, loss_bc, loss_pde, loss, r = loss_function(model, X, X_ic, y_ic, X_bc, y_bc, M)
        loss.backward()

        history["lbfgs"].append({
            "ic": loss_ic.item(),
            "bc": loss_bc.item(),
            "pde": loss_pde.item(),
            "general": loss.item(),
            "residual": r.item()
        })

        return loss

    start = time.time()
    for _ in range(lbfgs_params["epochs"] + 1):
        optimizer.step(closure)
    end = (time.time() - start) / 60

    history["time"]["lbfgs"] = end

    if save:
        torch.save(model.state_dict(), model_path)

    if save_history:
        json.dump(history, open(history_path, "w"))

    return history

def evaluate_model(model, X_tests, times, M, save=False, path="./results.json"):
    results = {"metrics": [], "plot_data": [], "times": times}

    for X_test in X_tests:
        x_test = X_test[:, 0:1].detach().cpu().numpy()
        t_test = X_test[:, 1:2].detach().cpu().numpy()
        y_pred = model(X_test).detach().cpu().numpy()
        y_true = solution(x_test, t_test, M)

        mse = np.mean((y_pred - y_true) ** 2).item()
        l2 = (np.linalg.norm(y_pred - y_true, 2) / np.linalg.norm(y_true, 2)).item()

        results["metrics"].append((mse, l2))
        results["plot_data"].append((x_test.squeeze().tolist(), y_pred.squeeze().tolist(), y_true.squeeze().tolist()))

    if save:
        json.dump(results, open(path, "w"))

    return results

def parse_history(history):
    adam_loss_ic = []
    adam_loss_bc =[]
    adam_loss_pde = []
    adam_loss = []
    adam_residual = []

    lbfgs_loss_ic = []
    lbfgs_loss_bc = []
    lbfgs_loss_pde = []
    lbfgs_loss = []
    lbfgs_residual = []

    for entry in history["adam"]:
        adam_loss_ic.append(entry["ic"])
        adam_loss_bc.append(entry["bc"])
        adam_loss_pde.append(entry["pde"])
        adam_loss.append(entry["general"])
        adam_residual.append(entry["residual"])

    for entry in history["lbfgs"]:
        lbfgs_loss_ic.append(entry["ic"])
        lbfgs_loss_bc.append(entry["bc"])
        lbfgs_loss_pde.append(entry["pde"])
        lbfgs_loss.append(entry["general"])
        lbfgs_residual.append(entry["residual"])

    return (adam_loss_ic, adam_loss_bc, adam_loss_pde, adam_loss, adam_residual,
            lbfgs_loss_ic, lbfgs_loss_bc, lbfgs_loss_pde, lbfgs_loss, lbfgs_residual)

def plot_flux(M):
    x = np.linspace(0, 1, 100)
    y = f(x, M)

    plt.figure(figsize=(6, 4), dpi=160)
    plt.suptitle("Fractional flow", fontsize=14)
    plt.plot(x, y)
    plt.show()

def plot_training_losses(train_title, train_data):
    adam_epochs = np.asarray(range(len(train_data["adam"])))
    lbfgs_epochs = np.asarray(range(len(train_data["lbfgs"])))
    adam_loss_ic, adam_loss_bc, adam_loss_pde, adam_loss, adam_residual, \
    lbfgs_loss_ic, lbfgs_loss_bc, lbfgs_loss_pde, lbfgs_loss, lbfgs_residual = parse_history(train_data)

    fig, axis = plt.subplots(2, 2, figsize=(10, 8), dpi=160)
    fig.suptitle(train_title, fontsize=14)
    axis[0][0].plot(adam_epochs, adam_loss_ic, label="loss_ic", color="blue")
    axis[0][0].plot(adam_epochs, adam_loss_bc, label="loss_bc", color="red")
    axis[0][0].plot(adam_epochs, adam_loss_pde, label="loss_pde", color="green")
    axis[0][0].plot(adam_epochs, adam_loss, label="loss", color="orange")
    axis[0][0].set_title("Adam Loss")
    axis[0][0].legend()
    axis[0][1].plot(adam_epochs, adam_residual, color="red")
    axis[0][1].set_title("Adam Residual")

    axis[1][0].plot(lbfgs_epochs, lbfgs_loss_ic, label="loss_ic", color="blue")
    axis[1][0].plot(lbfgs_epochs, lbfgs_loss_bc, label="loss_bc", color="red")
    axis[1][0].plot(lbfgs_epochs, lbfgs_loss_pde, label="loss_pde", color="green")
    axis[1][0].plot(lbfgs_epochs, lbfgs_loss, label="loss", color="orange")
    axis[1][0].set_title("L-BFGS Loss")
    axis[1][0].legend()
    axis[1][1].plot(lbfgs_epochs, lbfgs_residual, color="red")
    axis[1][1].set_title("L-BFGS Residual")
    plt.tight_layout()
    plt.show()

def plot_test_data(eval_title, eval_data):
    data = eval_data["plot_data"]
    times = eval_data["times"]

    fig, axis = plt.subplots(2, 2, figsize=(8, 6), dpi=160)
    fig.suptitle(eval_title, fontsize=14)
    axis[0][0].plot(data[0][0], data[0][1], label="predicted values", ls="--", color="red")
    axis[0][0].plot(data[0][0], data[0][2], label="exact values")
    axis[0][0].set_xlim(0, 1)
    axis[0][0].set_ylim(0, 1.1)
    axis[0][0].set_title(f"t={times[0]:.2f}")
    axis[0][0].legend()

    axis[0][1].plot(data[1][0], data[1][1], label="predicted values", ls="--", color="red")
    axis[0][1].plot(data[1][0], data[1][2], label="exact values")
    axis[0][1].set_xlim(0, 1)
    axis[0][1].set_ylim(0, 1.1)
    axis[0][1].set_title(f"t={times[1]:.2f}")
    axis[0][1].legend()

    axis[1][0].plot(data[2][0], data[2][1], label="predicted values", ls="--", color="red")
    axis[1][0].plot(data[2][0], data[2][2], label="exact values")
    axis[1][0].set_xlim(0, 1)
    axis[1][0].set_ylim(0, 1.1)
    axis[1][0].set_title(f"t={times[2]:.2f}")
    axis[1][0].legend()

    axis[1][1].plot(data[3][0], data[3][1], label="predicted values", ls="--", color="red")
    axis[1][1].plot(data[3][0], data[3][2], label="exact values")
    axis[1][1].set_xlim(0, 1)
    axis[1][1].set_ylim(0, 1.1)
    axis[1][1].set_title(f"t={times[3]:.2f}")
    axis[1][1].legend()
    plt.tight_layout()
    plt.show()

def plot_data(train_title, eval_title, train_path, eval_path):
    train_data = json.load(open(train_path, "r"))
    eval_data = json.load(open(eval_path, "r"))

    plot_training_losses(train_title, train_data)
    plot_test_data(eval_title, eval_data)
