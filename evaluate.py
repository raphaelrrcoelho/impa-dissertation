# File: evaluate.py
import json, sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env import CreditPortfolioEnv
from baselines import (strategy_cash_only, strategy_risk_free, strategy_equal_allocation,
                       strategy_greedy_expected_return, strategy_liability_matching_plan)

def evaluate_strategy(env, strategy_fn):
    """Simulate one strategy (either baseline or trained policy) for one scenario episode."""
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    # If strategy_fn is a Stable-Baselines policy (PPO model), it will have a predict() method
    is_trained_policy = hasattr(strategy_fn, "predict")
    while not done:
        if is_trained_policy:
            action, _ = strategy_fn.predict(obs, deterministic=True)
        else:
            action = strategy_fn(obs, env)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        # If bankrupt, can break early (episode terminated)
        if done:
            break
    final_cash = env.current_cash if not env.bankrupt else 0.0  # final remaining cash (0 if bankrupt)
    bankrupt = env.bankrupt
    return final_cash, bankrupt

def main():
    # Load config and model
    config_path = None
    model_path = None
    for arg in sys.argv[1:]:
        if arg.endswith(".json"):
            config_path = arg
        elif arg.endswith(".zip"):
            model_path = arg
    if config_path is None or model_path is None:
        print("Usage: python evaluate.py <config.json> <model.zip>")
        return
    with open(config_path, 'r') as f:
        config = json.load(f)
    # Load trained PPO model
    model = PPO.load(model_path)
    # Set up base scenario and evaluation environment
    base_scenario = config["scenario"]
    risk_scale = config.get("risk_scale", 1.0)
    shock = config.get("shock", False)
    liquidity_bonus = config.get("liquidity_bonus", False)
    base_scenario["liquidity_bonus"] = liquidity_bonus
    # Number of test scenarios
    n_test = config.get("n_test", 1000)
    np.random.seed(config.get("eval_seed", 123))
    # Results storage
    results = {
        "PPO": {"final_cash": [], "bankrupt": []},
        "Cash": {"final_cash": [], "bankrupt": []},
        "RiskFree": {"final_cash": [], "bankrupt": []},
        "LiabilityMatch": {"final_cash": [], "bankrupt": []},
        "Greedy": {"final_cash": [], "bankrupt": []},
        "Diversified": {"final_cash": [], "bankrupt": []}
    }
    # Simulate scenarios
    for i in range(n_test):
        # Generate a random scenario (similar to MultiScenarioEnv.reset)
        H = len(base_scenario["liabilities"])
        # Sample liabilities around base values
        liabilities = []
        for L in base_scenario["liabilities"]:
            L_val = np.random.uniform(0.8 * L, 1.2 * L)
            liabilities.append(float(L_val))
        if shock and H > 2:
            mid = H // 2
            liabilities[mid] *= 2.0
        init_cash = float(np.random.uniform(0.9 * base_scenario["initial_cash"], 1.1 * base_scenario["initial_cash"]))
        # Scale default probabilities
        asset_classes = []
        for asset in base_scenario["asset_classes"]:
            asset_copy = asset.copy()
            if asset_copy.get("default_prob", 0) > 0:
                asset_copy["default_prob"] = min(1.0, asset_copy["default_prob"] * risk_scale)
            asset_classes.append(asset_copy)
        scenario = {"liabilities": liabilities, "asset_classes": asset_classes, "initial_cash": init_cash}
        # Create a fresh environment for this scenario (no randomization inside, since scenario is fully specified now)
        env = CreditPortfolioEnv(scenario, liquidity_bonus=liquidity_bonus)
        # Evaluate PPO policy
        final_cash, bankrupt = evaluate_strategy(env, model)
        results["PPO"]["final_cash"].append(final_cash)
        results["PPO"]["bankrupt"].append(bankrupt)
        # Evaluate each baseline (reuse the same scenario for fairness)
        # Cash-only
        env = CreditPortfolioEnv(scenario, liquidity_bonus=liquidity_bonus)
        fc, bk = evaluate_strategy(env, strategy_cash_only)
        results["Cash"]["final_cash"].append(fc)
        results["Cash"]["bankrupt"].append(bk)
        # Risk-free
        env = CreditPortfolioEnv(scenario, liquidity_bonus=liquidity_bonus)
        fc, bk = evaluate_strategy(env, strategy_risk_free)
        results["RiskFree"]["final_cash"].append(fc)
        results["RiskFree"]["bankrupt"].append(bk)
        # Liability-matching deterministic plan
        env = CreditPortfolioEnv(scenario, liquidity_bonus=liquidity_bonus)
        fc, bk = evaluate_strategy(env, strategy_liability_matching_plan)
        results["LiabilityMatch"]["final_cash"].append(fc)
        results["LiabilityMatch"]["bankrupt"].append(bk)
        # Sharpe-ratio greedy
        env = CreditPortfolioEnv(scenario, liquidity_bonus=liquidity_bonus)
        fc, bk = evaluate_strategy(env, strategy_greedy_expected_return)
        results["Greedy"]["final_cash"].append(fc)
        results["Greedy"]["bankrupt"].append(bk)
        # Diversified equal mix
        env = CreditPortfolioEnv(scenario, liquidity_bonus=liquidity_bonus)
        fc, bk = evaluate_strategy(env, strategy_equal_allocation)
        results["Diversified"]["final_cash"].append(fc)
        results["Diversified"]["bankrupt"].append(bk)

    # Convert lists to numpy arrays for convenience
    for strat in results:
        results[strat]["final_cash"] = np.array(results[strat]["final_cash"], dtype=float)
        results[strat]["bankrupt"] = np.array(results[strat]["bankrupt"], dtype=bool)

    # Compute metrics
    def sharpe_ratio(x):
        # Sharpe ratio: mean(return) / std(return). Here treat final cash relative to initial as "return".
        # Use a small epsilon to avoid division by zero.
        return (np.mean(x) / (np.std(x) + 1e-9))
    def cvar(values, alpha=0.05):
        # CVaR_alpha: average of the worst alpha fraction of outcomes
        sorted_vals = np.sort(values)
        n_tail = max(1, int(alpha * len(sorted_vals)))
        return np.mean(sorted_vals[:n_tail])
    strategy_names = ["PPO", "Cash", "RiskFree", "LiabilityMatch", "Greedy", "Diversified"]
    print("Strategy Performance over {} test scenarios:".format(n_test))
    print("{:<15} {:>10} {:>12} {:>10} {:>10}".format("Strategy", "Avg Final", "Bankruptcy%", "Sharpe", "CVaR5%"))
    for strat in strategy_names:
        final_vals = results[strat]["final_cash"]
        bankrupts = results[strat]["bankrupt"]
        avg_final = np.mean(final_vals)
        bankrupt_pct = 100.0 * np.mean(bankrupts)
        sharpe = sharpe_ratio(final_vals)
        cvar5 = cvar(final_vals)
        print(f"{strat:<15} {avg_final:>10.2f} {bankrupt_pct:>12.1f} {sharpe:>10.3f} {cvar5:>10.2f}")
    # Statistical tests: PPO vs each baseline
    print("\nStatistical Comparison (PPO vs Baselines):")
    for strat in strategy_names[1:]:
        # Paired t-test on final cash
        tstat, pval = stats.ttest_rel(results["PPO"]["final_cash"], results[strat]["final_cash"])
        # McNemar test on bankruptcy outcomes
        # Compute contingency: b = PPO succeed & strat fail, c = PPO fail & strat succeed
        ppo_fail = results["PPO"]["bankrupt"]
        strat_fail = results[strat]["bankrupt"]
        b = np.sum((~ppo_fail) & (strat_fail))  # PPO success, baseline bankrupt
        c = np.sum((ppo_fail) & (~strat_fail))  # PPO bankrupt, baseline success
        # McNemar's test statistic
        mcnemar_stat = ((abs(b - c) - 1)**2) / float(b + c + 1e-9) if (b + c) > 0 else 0.0
        # p-value from chi-square with 1 df
        mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        print(f"PPO vs {strat}: t-test p={pval:.4f}, McNemar p={mcnemar_p:.4f}  (b={b}, c={c})")
    # Plot comparison (optional)
    if "--plot" in sys.argv:
        strategies = strategy_names
        avg_returns = [np.mean(results[s]["final_cash"]) for s in strategies]
        bankrupt_rates = [100.0 * np.mean(results[s]["bankrupt"]) for s in strategies]
        x = np.arange(len(strategies))
        width = 0.4
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        bars1 = ax1.bar(x - width/2, avg_returns, width, color='skyblue', label='Avg Final Cash')
        bars2 = ax2.bar(x + width/2, bankrupt_rates, width, color='salmon', label='Bankruptcy Rate (%)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(strategies)
        ax1.set_ylabel("Average Final Cash", color='blue')
        ax2.set_ylabel("Bankruptcy Rate (%)", color='red')
        plt.title("Strategy Performance Comparison")
        # Combine legends
        handles, labels = [], []
        for bar in [bars1, bars2]:
            handles.append(bar)
            labels.append(bar.get_label())
        plt.legend(handles, labels, loc="upper right")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
