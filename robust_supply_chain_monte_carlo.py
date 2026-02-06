"""
Monte Carlo Simulation for Robust Supply Chain Network Design
with CVaR-based Configuration Selection and Visualization
"""

import pandas as pd
import numpy as np
from pulp import *
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------

def load_data():
    manvar_costs = pd.read_excel("data/variable costs.xlsx", index_col=0)
    freight_costs = pd.read_excel("data/freight costs.xlsx", index_col=0)
    fixed_costs = pd.read_excel("data/fixed cost.xlsx", index_col=0)
    cap = pd.read_excel("data/capacity.xlsx", index_col=0)
    demand = pd.read_excel("data/demand.xlsx", index_col=0)

    var_cost = freight_costs / 1000 + manvar_costs
    return var_cost, fixed_costs, cap, demand


# ------------------------------------------------------------
# FIRST-STAGE MODEL (y free)
# ------------------------------------------------------------

def optimization_model(fixed_costs, var_cost, demand_values, cap):
    loc = ["USA", "GERMANY", "JAPAN", "BRAZIL", "INDIA"]
    size = ["LOW", "HIGH"]

    model = LpProblem("Plant_Location", LpMinimize)

    x = LpVariable.dicts(
        "x", [(i, j) for i in loc for j in loc], lowBound=0
    )
    y = LpVariable.dicts(
        "y", [(i, s) for s in size for i in loc], cat="Binary"
    )

    model += (
        lpSum([fixed_costs.loc[i, s] * y[(i, s)] * 1000 for s in size for i in loc])
        + lpSum([var_cost.loc[i, j] * x[(i, j)] for i in loc for j in loc])
    )

    for j in loc:
        model += lpSum(x[(i, j)] for i in loc) == demand_values[j]

    for i in loc:
        model += lpSum(x[(i, j)] for j in loc) <= lpSum(
            cap.loc[i, s] * y[(i, s)] * 1000 for s in size
        )

    model.solve()

    plant_order = [(i, s) for s in size for i in loc]
    plant_bool = [y[p].varValue for p in plant_order]

    return value(model.objective), plant_bool


# ------------------------------------------------------------
# SECOND-STAGE MODEL (y fixed)
# ------------------------------------------------------------

def second_stage_model(fixed_costs, var_cost, demand_values, cap, y_fixed):
    loc = ["USA", "GERMANY", "JAPAN", "BRAZIL", "INDIA"]
    size = ["LOW", "HIGH"]

    model = LpProblem("Evaluation", LpMinimize)

    x = LpVariable.dicts(
        "x", [(i, j) for i in loc for j in loc], lowBound=0
    )

    fixed_cost_term = lpSum(
        fixed_costs.loc[i, s] * y_fixed[(i, s)] * 1000
        for s in size for i in loc
    )

    model += fixed_cost_term + lpSum(
        var_cost.loc[i, j] * x[(i, j)] for i in loc for j in loc
    )

    for j in loc:
        model += lpSum(x[(i, j)] for i in loc) == demand_values[j]

    for i in loc:
        model += lpSum(x[(i, j)] for j in loc) <= lpSum(
            cap.loc[i, s] * y_fixed[(i, s)] * 1000 for s in size
        )

    model.solve()
    return value(model.objective)


# ------------------------------------------------------------
# DEMAND SCENARIOS
# ------------------------------------------------------------

def generate_demand_scenarios(base_demand, n, cv):
    np.random.seed(42)
    data = {}

    for k, mu in base_demand.items():
        sigma = cv * mu
        samples = np.random.normal(mu, sigma, n)
        data[k] = np.clip(samples, 0, None)

    return pd.DataFrame(data)


# ------------------------------------------------------------
# RISK METRICS
# ------------------------------------------------------------

def compute_cvar(costs, alpha=0.9):
    var_alpha = np.quantile(costs, alpha)
    cvar_alpha = costs[costs >= var_alpha].mean()
    return var_alpha, cvar_alpha


# ------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------

def plot_cost_distributions(cost_dict):
    plt.figure(figsize=(10, 6))
    for k, costs in cost_dict.items():
        plt.hist(costs, bins=20, alpha=0.4, label=f"C{k}")
    plt.xlabel("Total Cost")
    plt.ylabel("Frequency")
    plt.title("Cost Distribution per Candidate Configuration")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cost_distributions.png", dpi=150)
    plt.close()


def plot_cvar_bar(df_stats):
    plt.figure(figsize=(8, 5))
    plt.bar(df_stats["Configuration"], df_stats["CVaR_0.9"])
    plt.xlabel("Configuration")
    plt.ylabel("CVaR (0.9)")
    plt.title("CVaR₀.₉ Comparison Across Configurations")
    plt.tight_layout()
    plt.savefig("cvar_comparison.png", dpi=150)
    plt.close()


# ------------------------------------------------------------
# MAIN DRIVER
# ------------------------------------------------------------

def run_monte_carlo(n_scenarios=50, cv=0.5):

    var_cost, fixed_costs, cap, demand = load_data()

    loc = ["USA", "GERMANY", "JAPAN", "BRAZIL", "INDIA"]
    size = ["LOW", "HIGH"]

    base_demand = demand["Demand"].to_dict()
    df_scenarios = generate_demand_scenarios(base_demand, n_scenarios, cv)

    # ---------- First stage ----------
    df_bool = pd.DataFrame()
    for k in range(n_scenarios):
        cost, plant_bool = optimization_model(
            fixed_costs, var_cost, df_scenarios.iloc[k].to_dict(), cap
        )
        df_bool[k] = plant_bool

    df_bool = df_bool.astype(int)
    df_candidates = df_bool.T.drop_duplicates().T

    print(f"\nUnique candidate configurations: {df_candidates.shape[1]}")

    # ---------- Second stage ----------
    stats = []
    cost_map = {}

    for c in df_candidates.columns:
        y_vec = df_candidates[c].values
        y_fixed = {}
        idx = 0
        for s in size:
            for i in loc:
                y_fixed[(i, s)] = y_vec[idx]
                idx += 1

        costs = []
        for k in range(n_scenarios):
            cost = second_stage_model(
                fixed_costs, var_cost,
                df_scenarios.iloc[k].to_dict(),
                cap, y_fixed
            )
            costs.append(cost)

        costs = np.array(costs)
        var90, cvar90 = compute_cvar(costs)

        stats.append({
            "Configuration": c,
            "Mean": costs.mean(),
            "Std": costs.std(),
            "VaR_0.9": var90,
            "CVaR_0.9": cvar90
        })

        cost_map[c] = costs

    df_stats = pd.DataFrame(stats).sort_values("CVaR_0.9")

    # ---------- Plots ----------
    plot_cost_distributions(cost_map)
    plot_cvar_bar(df_stats)

    # ---------- Best configuration ----------
    best = df_stats.iloc[0]
    best_col = best["Configuration"]

    print("\n===== BEST CONFIGURATION (CVaR 0.9 OPTIMAL) =====")
    print(best)

    print("\nOpen plants:")
    y_best = df_candidates[best_col].values
    idx = 0
    for s in size:
        for i in loc:
            if y_best[idx] == 1:
                print(f"  - {i} – {s}")
            idx += 1

    return df_stats


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":
    run_monte_carlo(n_scenarios=50, cv=0.5)
