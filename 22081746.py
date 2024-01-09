import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Read data from the file
data = pd.read_csv('data6-1.csv', header=None)

# Assign 'Salary' as the column name
data.columns = ['Salary']

# Calculate mean annual salary
mean_salary = np.mean(data['Salary'])

# Create a histogram
plt.hist(data['Salary'], bins=30, density=True,
         alpha=0.7, label='Salary Distribution')

# Calculate the probability density function using a kernel density estimate
salary_range = np.linspace(data['Salary'].min(), data['Salary'].max(), 1000)
kde = gaussian_kde(data['Salary'])
plt.plot(salary_range, kde(salary_range), label='Probability Density Function')

# Calculate the fraction of population with salaries between mean and 1.25 times the mean
lower_bound = mean_salary
upper_bound = 1.25 * mean_salary
fraction_between = np.sum((data['Salary'] >= lower_bound) & (
    data['Salary'] <= upper_bound)) / len(data)

# Plot mean and fraction on the graph
plt.axvline(mean_salary, color='red', linestyle='--',
            label=f'Mean Salary: {mean_salary:.2f}')
plt.text(mean_salary, plt.ylim()[1]*0.9,
         f'Mean Salary: {mean_salary:.2f}', color='red')
plt.text(upper_bound, plt.ylim()[
         1]*0.5, f'Fraction(X)-(W~ to 1.25W~): {fraction_between:.2f}', color='green')

# Set labels, title, and legend
plt.xlabel('Salary')
plt.ylabel('Probability Density')
plt.title('Salary Distribution and Probability Density Function')
plt.xlim(0,plt.xlim()[1])
plt.ylim(0,plt.ylim()[1])
plt.legend()

# Show the plot
plt.show()
