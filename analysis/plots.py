import pandas as pd
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv('analysis/data/joint_training.csv')

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(data['Step'], data['Attacker Reward'], label='Attacker')
plt.plot(data['Step'], data['Defender Reward'], label='Defender')

# Add vertical lines at steps 20, 40, 60
for x in [0, 20, 40, 60, 80]:
    plt.axvline(x=x, color='gray', linestyle=':', alpha=0.5)

# Add labels and legend
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Attacker vs Defender Rewards Over Time')
plt.legend()

# Save and show
plt.tight_layout()
plt.savefig('analysis/plots/training_rewards.png')
plt.show()