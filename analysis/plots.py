import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('analysis/data/joint_training.csv')

plt.figure(figsize=(3.5, 2.5))


plt.plot(
    data['Step'],
    data['Defender Reward'],
    label='Defender',
    linewidth=2.0
)

plt.plot(
    data['Step'],
    data['Attacker Reward'],
    label='Attacker',
    linewidth=2.0
)


# Optional: keep only key vertical markers
for x in [20, 40, 60]:
    plt.axvline(x=x, linestyle=':', alpha=0.3)

plt.xlabel('Step', fontsize=9)
plt.ylabel('Reward', fontsize=9)

plt.legend(
    fontsize=8,
    frameon=False,
    loc='center right',
    bbox_to_anchor=(1.0, 0.65)
)

plt.tick_params(axis='both', labelsize=8)
plt.grid(alpha=0.2)

plt.ylim(-.05, 1.05)

plt.tight_layout()
plt.savefig(
    'analysis/plots/joint_training_rewards.png',
    dpi=300,
    bbox_inches='tight'
)


plt.show()
