
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Import data from the specified file
data = pd.read_csv('data6-1.csv', header=None)

# Assign 'Salary' as the column header
data.columns = ['Salary']

# Compute the average salary
mean_salary = np.mean(data['Salary'])

# Generate a histogram representing salary distribution
plt.hist(data['Salary'], bins=30, density=True, color='Yellow',
         edgecolor='black', alpha=0.7, label='Salary Distribution')

# Create a probability density function using kernel density estimation
salary_range = np.linspace(data['Salary'].min(), data['Salary'].max(), 1000)
kde = gaussian_kde(data['Salary'])
plt.plot(salary_range, kde(salary_range), color='Black',
         label='Probability Density Function')

# Calculate the proportion of the population with salaries between the mean and 1.25 times the mean
lower_bound = mean_salary
upper_bound = 1.25 * mean_salary
fraction_between = np.sum((data['Salary'] >= lower_bound) & (
    data['Salary'] <= upper_bound)) / len(data)

# Define a function to obtain statistical descriptors of the dataset


def getStatisticalDescription(data):
    median = np.median(data)
    mode = data.mode()
    std = data.std()
    kurtosis = data.kurtosis()
    skewness = data.skew()
    data_range = max(data) - min(data)
    return mode, median, std, kurtosis, skewness, data_range, max(data), min(data)


# Obtain statistical descriptors
mode, median, std, kurtosis, skewness, data_range, max_val, min_val = getStatisticalDescription(
    data['Salary'])

# Display the statistical descriptors
print(f"mode: {mode}, median: {median}, std: {std}, kurtosis: {kurtosis}, skewness: {skewness}, range: {data_range}, max: {max_val}, min: {min_val}")

# Plot the mean and fraction on the graph
plt.axvline(mean_salary, color='red', linestyle='--',
            label=f'Mean Salary')
plt.text(mean_salary, plt.ylim()[1]*0.8,
         f'Mean Salary: {mean_salary:.2f}', color='red')
plt.text(upper_bound, plt.ylim()[
         1]*0.65, f'Fraction(X): {fraction_between:.2f}', color='black')

# Set labels, title, and legend
plt.xlabel('Salary')
plt.ylabel('Probability Density')
plt.title('Salary Distribution and Probability Density Function')
plt.xlim(0, plt.xlim()[1])
plt.ylim(0, plt.ylim()[1])
plt.legend()
plt.tight_layout()

# Display the plot
plt.show()
