# File: env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class CreditPortfolioEnv(gym.Env):
    """
    Gymnasium Environment for multi-period Asset-Liability Management.
    The agent must allocate cash among assets to meet liabilities over a horizon.
    """
    def __init__(self, scenario, liquidity_bonus=False):
        super().__init__()
        # Unpack scenario parameters
        self.liabilities = scenario["liabilities"]             # list of liabilities [L0, L1, ..., L_{H-1}]
        self.asset_classes = scenario["asset_classes"]         # list of asset dicts: each has default_prob, payoff_factor, maturity, etc.
        self.initial_cash = scenario.get("initial_cash", 0.0)  # starting cash
        self.horizon = len(self.liabilities)                   # number of time steps
        self.liquidity_bonus = liquidity_bonus

        # Define action space: allocation fractions for each asset class (length = N assets).
        # Fractions must be between 0 and 1; if their sum is less than 1, the remainder stays as cash.
        N = len(self.asset_classes)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(N,), dtype=np.float32)
        # Define observation space: [current_cash, current_time_index, next_liability].
        # (We ensure bounds are reasonable upper estimates for cash and liabilities.)
        obs_low  = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1e9, self.horizon, 1e9], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(3,), dtype=np.float32)

        # Internal state
        self.current_step = 0
        self.current_cash = 0.0
        self.pending_investments = []  # list of dicts for investments: {"maturity_step": t, "amount": x, "default_prob": p, "payoff_factor": f, ...}
        self.bankrupt = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_cash = self.initial_cash
        self.bankrupt = False
        self.pending_investments.clear()
        # Initial observation includes current cash, current time (0), and the liability due at step 0.
        obs = np.array([self.current_cash, float(self.current_step), self.liabilities[0]], dtype=np.float32)
        return obs, {}

    def step(self, action):
        if self.current_step >= self.horizon:
            # Episode already ended
            return None, 0.0, True, {}

        # 1. Realize payoffs for investments maturing at this step
        payoff = 0.0
        for inv in list(self.pending_investments):
            if inv["maturity_step"] == self.current_step:
                # Investment matures now
                if np.random.rand() < inv["default_prob"]:
                    # Default occurs: receive recovery (assume 0 for simplicity)
                    payoff_amount = inv.get("recovery_rate", 0.0) * inv["amount"]
                else:
                    # No default: receive principal * payoff_factor (e.g., 1 + interest)
                    payoff_amount = inv["amount"] * inv["payoff_factor"]
                payoff += payoff_amount
                # Remove this investment from pending list
                self.pending_investments.remove(inv)
        # Add any payoff to current cash
        self.current_cash += payoff

        # 2. Pay the current liability
        liability = self.liabilities[self.current_step]
        self.current_cash -= liability
        if self.current_cash < 0:
            # Bankruptcy: liability could not be fully paid
            self.bankrupt = True
            reward = -1000.0  # large negative reward for bankruptcy
            terminated = True
            # Observation in bankruptcy (could mark cash as 0 and show remaining liability)
            obs = np.array([0.0, float(self.current_step), liability], dtype=np.float32)
            info = {"bankrupt": True, "step": self.current_step}
            return obs, reward, terminated, info

        # 3. Allocate surplus (current_cash) according to action
        surplus = self.current_cash
        alloc = np.clip(np.array(action, dtype=np.float32), 0.0, 1.0)
        if alloc.sum() > 0:
            if self.liquidity_bonus:
                # If liquidity bonus is on, allow partial allocation (do not force full investment).
                if alloc.sum() > 1.0:
                    alloc = alloc / alloc.sum()  # cap total allocation at 100%
            else:
                # Without liquidity bonus, normalize any positive action to invest entire surplus.
                alloc = alloc / alloc.sum()
        # If alloc.sum() == 0, no investments (keep all cash)
        invested_amount = 0.0
        for i, fraction in enumerate(alloc):
            amount = fraction * surplus
            if amount <= 0:
                continue
            # Determine maturity of the asset and schedule payoff
            asset = self.asset_classes[i]
            maturity = asset["maturity"]
            target_step = self.current_step + maturity
            # Only invest if payoff comes before or at horizon (else skip, as it won't help pay liabilities in time)
            if target_step > self.horizon:
                continue
            inv = {
                "maturity_step": target_step,
                "amount": amount,
                "default_prob": asset["default_prob"],
                "payoff_factor": asset["payoff_factor"]
            }
            # Include recovery_rate if provided in asset definition
            if "recovery_rate" in asset:
                inv["recovery_rate"] = asset["recovery_rate"]
            self.pending_investments.append(inv)
            invested_amount += amount
        # Deduct invested amount from cash (any remainder stays as cash on hand)
        self.current_cash -= invested_amount

        # 4. Compute reward for this step
        reward = 0.0
        terminated = False
        # If this was the last period, episode terminates and reward includes final remaining cash
        if self.current_step == self.horizon - 1:
            reward = self.current_cash  # final reward: leftover cash after last liability
            terminated = True
        else:
            # Intermediate reward: optionally give a small bonus for liquidity remaining (cash not invested)
            if self.liquidity_bonus:
                reward += 0.01 * self.current_cash  # small bonus for cash retention

        # 5. Prepare next observation
        self.current_step += 1
        if terminated:
            # Episode done, next liability is not applicable
            next_liability = 0.0
        else:
            next_liability = self.liabilities[self.current_step]
        obs = np.array([self.current_cash, float(self.current_step), next_liability], dtype=np.float32)
        info = {}
        return obs, reward, terminated, info
