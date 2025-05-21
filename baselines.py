# File: baselines.py
import numpy as np
import cvxpy as cp

# Index 0 is assumed to be the risk-free asset in asset_classes
RISK_FREE_INDEX = 0

def strategy_cash_only(observation, env):
    """Allocate 0% to all assets (keep all cash)."""
    n = env.action_space.shape[0]
    action = np.zeros(n, dtype=np.float32)
    return action

def strategy_risk_free(observation, env):
    """Allocate 100% of surplus to the risk-free asset."""
    n = env.action_space.shape[0]
    action = np.zeros(n, dtype=np.float32)
    action[RISK_FREE_INDEX] = 1.0  # all surplus to risk-free bond
    return action

def strategy_equal_allocation(observation, env):
    """Allocate surplus equally across all asset classes."""
    n = env.action_space.shape[0]
    action = np.ones(n, dtype=np.float32) / n
    return action

def strategy_greedy_expected_return(observation, env):
    """Allocate all surplus to the single asset with highest expected payoff (payoff_factor * survival_prob)."""
    expected_returns = []
    for asset in env.asset_classes:
        # Calculate expected payoff factor = (1 - default_prob) * payoff_factor
        exp_payoff = (1.0 - asset["default_prob"]) * asset["payoff_factor"]
        expected_returns.append(exp_payoff)
    expected_returns = np.array(expected_returns, dtype=np.float32)
    best_idx = int(np.argmax(expected_returns))
    n = env.action_space.shape[0]
    action = np.zeros(n, dtype=np.float32)
    action[best_idx] = 1.0  # allocate all to the best asset
    return action

def strategy_liability_matching_plan(observation, env):
    """Follow a precomputed deterministic plan (LP solution) for investing to meet liabilities."""
    # If this is the first step, compute the optimal plan ignoring default risk.
    if env.current_step == 0:
        plan = solve_deterministic_plan(env.liabilities, env.asset_classes, env.initial_cash)
        if plan is None:
            # No feasible plan (initial cash insufficient even in expectation) -> default to risk-free strategy
            env._planned_actions = [strategy_risk_free]*env.horizon
        else:
            # Convert plan (amounts) to allocation fractions for each step
            planned_fractions = []
            for t in range(env.horizon):
                # Surplus expected at step t in the plan = sum of investments at t (since plan uses all available)
                planned_surplus = plan[t, :].sum()
                if planned_surplus <= 0:
                    # No investment at this step (likely end or no surplus)
                    frac = np.zeros(plan.shape[1], dtype=np.float32)
                else:
                    frac = plan[t, :] / planned_surplus
                planned_fractions.append(frac.astype(np.float32))
            # Store the sequence of planned allocation fractions for each step
            env._planned_actions = planned_fractions
    # Use the precomputed fraction for the current step (if available)
    if hasattr(env, "_planned_actions") and env.current_step < len(env._planned_actions):
        action = env._planned_actions[env.current_step]
    else:
        # Fallback to no allocation if something goes wrong
        n = env.action_space.shape[0]
        action = np.zeros(n, dtype=np.float32)
    return action

def solve_deterministic_plan(liabilities, asset_classes, initial_cash):
    """
    Solve a linear program (via CVXPY) to maximize final cash surplus, 
    subject to covering all liabilities, assuming no default risk (dp=0).
    Returns a matrix plan[t,i] = amount to invest in asset i at time t.
    """
    H = len(liabilities)
    N = len(asset_classes)
    # Decision variables: x[t][i] = amount invested in asset i at time t
    x = cp.Variable((H, N), nonneg=True)
    # Variable for final leftover cash after last liability
    leftover_final = cp.Variable(nonneg=True)

    constraints = []
    # Cash flow dynamics constraints for each time step
    for t in range(H):
        # Calculate inflow (cash available) at time t
        if t == 0:
            inflow_t = initial_cash
        else:
            # Inflow at t = sum of payoffs from all investments made at earlier times that mature at t
            inflow_terms = []
            for j in range(t):
                for i, asset in enumerate(asset_classes):
                    # Only consider assets with no default (dp=0 assumption)
                    # If an asset has maturity that brings payoff at time t:
                    if j + asset["maturity"] == t:
                        inflow_terms.append(x[j, i] * asset["payoff_factor"])
            if inflow_terms:
                inflow_t = cp.sum(inflow_terms)
            else:
                inflow_t = 0
        # Outflow at t = liability at t + investments made at t (surplus allocated)
        if t < H - 1:
            # For steps before final, all surplus can be invested
            outflow_t = liabilities[t] + cp.sum(x[t, :])
        else:
            # At last liability step, disallow any new investments (they wouldn't mature in time)
            outflow_t = liabilities[t]
            constraints.append(x[t, :] == 0)
            # Define leftover_final as any cash remaining after last liability
            constraints.append(leftover_final == inflow_t - liabilities[t])
        # Require that inflow_t covers outflow_t (no shortfall at time t)
        constraints.append(inflow_t >= outflow_t)
    # Objective: maximize final leftover cash
    objective = cp.Maximize(leftover_final)
    prob = cp.Problem(objective, constraints)
    try:
        result = prob.solve(solver=cp.ECOS, verbose=False)
    except Exception as e:
        result = None
    if result is None or prob.status not in ["optimal", "optimal_infeasible", "optimal_inaccurate", "optimal"]:
        return None
    # Extract plan as a NumPy array
    plan = np.array(x.value, dtype=float)
    return plan
