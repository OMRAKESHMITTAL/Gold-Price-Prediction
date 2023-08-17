import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge

# Read the CSV file
df = pd.read_csv('FINAL_USO.csv')

# Convert date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Calculate the number of days since a reference date
reference_date = datetime(2000, 1, 1)
df['days_since_reference'] = (df['Date'] - reference_date).dt.days

# Calculate the correlation matrix
correlation_matrix = df.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Plot using Matplotlib and Seaborn
plot_columns = df.drop(['Date', 'days_since_reference'], axis=1).columns

plt.figure(figsize=(10, 6))
for column in plot_columns:
    sns.lineplot(x='days_since_reference', y=column, data=df, label=column)

plt.title('Time Series of Different Variables')
plt.xlabel('Days Since Reference Date')
plt.ylabel('Value')
plt.legend()
plt.show()

dfi = df.set_index('Date')

test_size = int(0.3 * len(dfi))
train_size = len(dfi) - test_size

training_x = dfi[:train_size]

testing_x = dfi[train_size:].drop(['Adj Close', 'Close'], axis=1)
testing_y = list(dfi[train_size:].drop('Close', axis=1)['Adj Close'].values)

# Initialize Ridge model
best_model = Ridge(alpha=1.0, fit_intercept=True, max_iter=None, random_state=1858, solver='auto', tol=0.001)

# Fit the model
best_model_fit = best_model.fit(training_x.drop(['Adj Close', 'Close'], axis=1), training_x['Adj Close'])

# Predict using the model
best_model_predict = list(best_model_fit.predict(testing_x))

results_df = pd.DataFrame()
results_df['Predictions'] = best_model_predict
results_df['Actual'] = testing_y

# Plot Feature Importance using Seaborn
coefs = list(abs(best_model.coef_))
cols = list(training_x.drop(['Close', 'Adj Close'], axis=1).columns)

var_importance = pd.DataFrame({'cols': cols, 'coefs': coefs})
var_importance = var_importance.sort_values('coefs', ascending=False)[:10]

plt.figure(figsize=(10, 6))
sns.barplot(x='coefs', y='cols', data=var_importance)
plt.title('Top 10 Feature Importance')
plt.xlabel('Coefficient Absolute Value')
plt.ylabel('Feature')
plt.show()

# Input for prediction
input_date_str = input("Enter date (YYYY-MM-DD): ")
input_date = datetime.strptime(input_date_str, '%Y-%m-%d')
days_since_reference = (input_date - reference_date).days

input_data = {
    'Open': float(input("Enter open price: ")),
    'High': float(input("Enter high: ")),
    'Low': float(input("Enter low: ")),
    'Volume': float(input("Enter volume: ")),
    'days_since_reference': days_since_reference
}

# Create a DataFrame from the input data
input_df = pd.DataFrame([input_data])

# Predict using the model
prediction = best_model_fit.predict(input_df)

# Print the prediction
print("Predicted Adj Close Price:", prediction[0])
