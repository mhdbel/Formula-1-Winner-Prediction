import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_matrix(data):
    """
    Plot a correlation matrix for numeric features.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.show()

def plot_win_distribution(data):
    """
    Plot the distribution of wins by driver.
    """
    plt.figure(figsize=(12, 6))
    sns.countplot(data=data, x='DriverNumber', hue='Win')
    plt.xticks(rotation=90)
    plt.title('Wins by Driver')
    plt.show()

if __name__ == '__main__':
    # Example usage
    processed_data = pd.read_csv('data/processed_data/processed_canadian_gp.csv')
    plot_correlation_matrix(processed_data)
    plot_win_distribution(processed_data)