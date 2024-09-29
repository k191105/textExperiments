import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from scipy.stats import ttest_rel

textType = 'blog'
data = pd.read_csv(f'entropyValues/entropy_{textType}.csv')

def chi_squared(data):

    count_human_gt_ai = sum(data['Entropy Human'] > data['Entropy AI'])
    count_ai_gt_human = sum(data['Entropy AI'] > data['Entropy Human'])
    total_counts = count_human_gt_ai + count_ai_gt_human

    count_data = pd.DataFrame({
        'Comparison': ['Human > AI', 'AI > Human'],
        'Observed': [count_human_gt_ai, count_ai_gt_human]
    })

    expected_counts = [total_counts / 2, total_counts / 2]
    chi2_statistic, p_value = chisquare(f_obs=count_data['Observed'], f_exp=expected_counts)

    print("Counts of Entropy Comparisons (excluding ties):")
    print(count_data)
    print(f"\nChi-Squared Statistic: {chi2_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")

    plt.figure(figsize=(8, 6))
    sns.barplot(
        x='Comparison',
        y='Observed',
        data=count_data,
        palette='viridis'
    )
    plt.title('Number of Times Entropy Comparison Holds (Excluding Ties)')
    plt.xlabel('Entropy Comparison')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


def paired_t_test(data):
    # Calculate the difference between human and AI entropies
    entropy_human = data['Entropy Human']
    entropy_ai = data['Entropy AI']

    t_statistic, p_value = ttest_rel(entropy_human, entropy_ai)

    print(f"Paired T-Test Statistic: {t_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(entropy_human, label='Entropy Human', color='blue', alpha=0.7)
    plt.plot(entropy_ai, label='Entropy AI', color='red', alpha=0.7)
    plt.title('Entropy Values: Human vs AI')
    plt.xlabel('Instance')
    plt.ylabel('Entropy')
    plt.legend()
    plt.tight_layout()
    plt.show()

chi_squared(data)
paired_t_test(data)
