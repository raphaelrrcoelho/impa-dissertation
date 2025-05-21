# main.py
import argparse
import numpy as np

from env import CreditPortfolioEnv
from baselines import (
    strategy_risk_free,
    strategy_equal_allocation,
    strategy_greedy_expected_return
)
from train import (
    train_sac_agent,        # Single-scenario training
    train_sac_agent_multi   # Multi-scenario training
)
from evaluate import evaluate_strategies
from plot_utils import (
    plot_training_rewards,
    plot_strategy_comparison,
    plot_cash_vs_liability
)

from stable_baselines3 import SAC

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", 
                        help="Train a new model on a single scenario (train_sac_agent).")
    parser.add_argument("--multi", action="store_true", 
                        help="Train a new model on multiple random scenarios (train_sac_agent_multi).")
    parser.add_argument("--timesteps", type=int, default=200000,
                        help="Number of timesteps to train for.")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Number of episodes per scenario-strategy during evaluation.")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Global random seed for training/eval.")
    args = parser.parse_args()

    model_file = "sac_credit_portfolio.zip"
    model = None

    # CHOOSE TRAINING MODE OR LOAD EXISTING
    if args.train and args.multi:
        print("[MAIN] Both --train and --multi are set, defaulting to multi-scenario training.")
        model = train_sac_agent_multi(
            total_timesteps=args.timesteps,
            model_save_path=model_file,
            seed=args.seed
        )
    elif args.train:
        print("[MAIN] Training a new model on a single scenario (train_sac_agent).")
        model = train_sac_agent(
            total_timesteps=args.timesteps,
            model_save_path=model_file
        )
    elif args.multi:
        print("[MAIN] Training a new model on multiple random scenarios (train_sac_agent_multi).")
        model = train_sac_agent_multi(
            total_timesteps=args.timesteps,
            model_save_path=model_file,
            seed=args.seed
        )
    else:
        print("[MAIN] Loading existing model:", model_file)
        model = SAC.load(model_file)

    # AFTER TRAINING/LOADING, EVALUATE ON 100 SCENARIOS
    print("[MAIN] Starting evaluation on 100 scenarios.")
    results = evaluate_strategies(model_file=model_file, 
                                  n_episodes=args.eval_episodes, 
                                  seed=args.seed)
    # Print the structured report
    print("\nEVALUATION RESULTS:")
    for row in results:
        # row keys: scenario_index, scenario_label, strategy, bankrupt_pct, avg_return, median_return
        print(row)

    # Example of plotting comparison if you want a single aggregated chart. 
    # Note: 'plot_strategy_comparison' typically expects a smaller subset or one scenario's summary.
    # You can adapt the function or filter 'results' as needed.
    # E.g. if you want to group by strategy, you might need a custom aggregator. 
    # We'll just do a simple usage:
    plot_strategy_comparison(results)  # might show a bar chart for bankrupt% vs. return

    # DEMO: if you want to do a single scenario run with the RL agent to log cash vs liability
    # We'll pick a scenario from your train or define a small scenario
    single_demo_scenario = {
        "liabilities": [10, 15, 20, 20, 15, 10],
        "asset_classes": [
            {"name": "RiskFree", "maturity":1, "default_prob":0.0,  "payoff_factor":1.001},
            {"name": "HG_3m",   "maturity":3, "default_prob":0.01, "payoff_factor":1.02},
            {"name": "HG_6m",   "maturity":6, "default_prob":0.02, "payoff_factor":1.05},
            {"name": "MG_3m",   "maturity":3, "default_prob":0.05, "payoff_factor":1.08},
            {"name": "MG_6m",   "maturity":6, "default_prob":0.10, "payoff_factor":1.15},
            {"name": "LG_1m",   "maturity":1, "default_prob":0.10, "payoff_factor":1.10}
        ],
        "initial_cash": 30.0
    }
    env = CreditPortfolioEnv(single_demo_scenario)
    obs, _ = env.reset(seed=args.seed)
    cash_hist, liab_hist = [], []
    done = False
    while not done:
        cash_hist.append(env.current_cash)
        # If current_month < horizon, record the liability
        if env.current_month < env.horizon:
            liab_hist.append(env.liabilities[env.current_month])
        else:
            liab_hist.append(0.0)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            # final step
            cash_hist.append(env.current_cash)
            if env.current_month < env.horizon:
                liab_hist.append(env.liabilities[env.current_month])
            break
    plot_cash_vs_liability(cash_hist, liab_hist)


if __name__ == "__main__":
    main()
