# File: train.py
import json, sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env import CreditPortfolioEnv

class MultiScenarioEnv(CreditPortfolioEnv):
    """
    Extends CreditPortfolioEnv to randomly generate a new scenario at each reset.
    This allows training across many random scenarios (liabilities and initial cash vary).
    """
    def __init__(self, base_scenario, risk_scale=1.0, shock=False):
        # Initialize with a dummy scenario; will override in reset()
        super().__init__(base_scenario, liquidity_bonus=base_scenario.get("liquidity_bonus", False))
        self.base_scenario = base_scenario
        self.risk_scale = risk_scale
        self.shock = shock
        # Random generator for scenario variations
        self.np_random = np.random.default_rng()

    def reset(self, seed=None, options=None):
        # If a seed is provided, use it for reproducibility
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        # Randomize liabilities: e.g., sample liabilities from a uniform or other distribution
        H = len(self.base_scenario["liabilities"])
        # For example, draw each liability from a range [L_min, L_max] around the base values
        base_liabs = self.base_scenario["liabilities"]
        # Here we use a simple approach: uniform random around +/-20% of base liability
        liabilities = []
        for L in base_liabs:
            L_min = 0.8 * L
            L_max = 1.2 * L
            liabilities.append(float(self.np_random.uniform(L_min, L_max)))
        # If shock enabled, apply a spike at mid-horizon
        if self.shock and H > 2:
            mid = H // 2
            liabilities[mid] *= 2.0  # double the liability at midpoint as a shock
        # Randomize initial cash (e.g., within some range of base initial cash)
        base_cash = self.base_scenario.get("initial_cash", 0.0)
        init_cash = float(self.np_random.uniform(0.9 * base_cash, 1.1 * base_cash))
        # Copy asset classes and scale default probabilities by risk_scale
        asset_classes = []
        for asset in self.base_scenario["asset_classes"]:
            asset_copy = asset.copy()
            # Scale default probability (risk-free remains 0)
            if "default_prob" in asset_copy and asset_copy["default_prob"] > 0:
                asset_copy["default_prob"] = min(1.0, asset_copy["default_prob"] * self.risk_scale)
            asset_classes.append(asset_copy)
        # Update the scenario for this episode
        self.liabilities = liabilities
        self.horizon = len(liabilities)
        self.initial_cash = init_cash
        self.asset_classes = asset_classes
        # Ensure the observation_space time dimension high bound covers the new horizon
        self.observation_space.high[1] = self.horizon
        # Call the base reset to set up initial state
        return super().reset(seed=seed, options=options)

def main():
    # Load config
    config_path = None
    for arg in sys.argv[1:]:
        if arg.endswith(".json"):
            config_path = arg
    if config_path is None:
        print("Usage: python train.py <config.json>")
        return
    with open(config_path, 'r') as f:
        config = json.load(f)
    # Base scenario template (liabilities list, asset classes, initial cash)
    # This can be part of config; here we assume it's included in config under "scenario".
    base_scenario = config["scenario"]
    # Environment toggles
    risk_scale = config.get("risk_scale", 1.0)
    shock = config.get("shock", False)
    liquidity_bonus = config.get("liquidity_bonus", False)
    base_scenario["liquidity_bonus"] = liquidity_bonus

    # Curriculum setup
    curriculum = config.get("curriculum", False)
    total_timesteps = config.get("total_timesteps", 200000)
    seed = config.get("seed", 42)
    net_size = config.get("network_size", "small")
    # Set network architecture for policy
    if net_size == "large":
        policy_kwargs = {"net_arch": [128, 128]}
    else:
        policy_kwargs = {"net_arch": [64, 64]}
    # Initialize model (will replace environment if doing curriculum)
    model = None

    if curriculum:
        # Phase 1: lower risk (e.g., half risk_scale, no shock)
        phase1_env = Monitor(MultiScenarioEnv(base_scenario, risk_scale=0.5 * risk_scale, shock=False))
        model = PPO("MlpPolicy", phase1_env, seed=seed, verbose=1, policy_kwargs=policy_kwargs)
        print("Curriculum Phase 1: Training on risk_scale={} for {} timesteps...".format(0.5 * risk_scale, total_timesteps // 3))
        model.learn(total_timesteps // 3)
        # Phase 2: normal risk
        phase2_env = Monitor(MultiScenarioEnv(base_scenario, risk_scale=risk_scale, shock=False))
        model.set_env(phase2_env)
        print("Curriculum Phase 2: Training on risk_scale={} for {} timesteps...".format(risk_scale, total_timesteps // 3))
        model.learn(total_timesteps // 3)
        # Phase 3: high risk or introduce shock
        phase3_env = Monitor(MultiScenarioEnv(base_scenario, risk_scale=1.5 * risk_scale, shock=shock))
        model.set_env(phase3_env)
        print("Curriculum Phase 3: Training on risk_scale={}{}, for {} timesteps...".format(
            1.5 * risk_scale, " with shock" if shock else "", total_timesteps - 2*(total_timesteps // 3)))
        model.learn(total_timesteps - 2*(total_timesteps // 3))
    else:
        # Single-phase training on specified risk level (and shock if any)
        env = Monitor(MultiScenarioEnv(base_scenario, risk_scale=risk_scale, shock=shock))
        model = PPO("MlpPolicy", env, seed=seed, verbose=1, policy_kwargs=policy_kwargs)
        print("Training PPO on risk_scale={}{} for {} timesteps...".format(risk_scale, " with shock" if shock else "", total_timesteps))
        model.learn(total_timesteps)

    # Save trained model
    model_path = config.get("model_path", "models/ppo_model.zip")
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
