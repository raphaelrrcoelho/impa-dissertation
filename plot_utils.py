"""
plot_utils.py: Utility functions for plotting (training rewards, strategy comparison).
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_rewards(reward_history, window=50):
    """
    Plot moving average of training rewards.
    reward_history: list of episode rewards (floats).
    """
    if window > 1 and len(reward_history) >= window:
        cumsum = np.cumsum(np.insert(reward_history, 0, 0)) 
        smoothed = (cumsum[window:] - cumsum[:-window]) / float(window)
        plt.plot(smoothed, label=f"Mean Reward (window {window})")
    else:
        plt.plot(reward_history, label="Episode Reward")
    plt.title("Training Reward Trajectory")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

def plot_strategy_comparison(results):
    """
    Plot average return and bankruptcy rate for each strategy.
    results: list of dicts with keys 'strategy', 'avg_final_cash', 'bankrupt_%'.
    """
    strategies = []
    avg_returns = []
    bankrupt_rates = []
    for res in results:
        strategies.append(res["strategy"])
        avg_returns.append(res["avg_final_cash"])
        bankrupt_rates.append(res["bankrupt_%"])
    x = np.arange(len(strategies))
    width = 0.4

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    bars1 = ax1.bar(x - width/2, avg_returns, width, color='skyblue', label='Avg Return')
    bars2 = ax2.bar(x + width/2, bankrupt_rates, width, color='salmon', label='Bankruptcy %')
    ax1.set_xlabel("Strategy")
    ax1.set_ylabel("Average Final Cash")
    ax2.set_ylabel("% Episodes Bankrupt")
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies)
    plt.title("Strategy Performance Comparison")
    # Add combined legend
    handles = [bars1, bars2]
    labels = [h.get_label() for h in handles]
    plt.legend(handles, labels, loc='upper left')
    plt.show()


def plot_box(df):
    plt.figure(figsize=(8,4))
    df.boxplot(column="final_cash", by="strategy")
    plt.title("Distribution of Final Cash by Strategy")
    plt.suptitle("")
    plt.ylabel("Final Cash")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "final_cash_boxplot.png"))
    plt.close()