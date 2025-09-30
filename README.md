# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('powerconsumption.csv')

# Use the 'Temperature' column (remove the leading space)
close_prices = data['Temperature'].dropna()

plt.rcParams['figure.figsize'] = [10, 7.5]

# Simulate ARMA(1,1) Process
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Step 2: Load the gold dataset
file_path = 'Gold Price Prediction.csv'  # Adjust the file path as necessary
data = pd.read_csv(file_path)

# Display column names to find the price column
print("Column names in the dataset:", data.columns)

# Step 3: Use the 'Price Today' column for the ARMA model
# Make sure to replace 'Price Today' with the correct column name if needed
price_data = data['Price Today'].dropna()  # Drop NaN values if any

# Step 4: Set up matplotlib settings for figure size
plt.rcParams['figure.figsize'] = [10, 7.5]

# Step 5: Define an ARMA(1,1) process with coefficients ar1 and ma1
# These coefficients are arbitrary; you can adjust them based on your analysis
ar1 = np.array([1, -0.5])  # AR coefficient
ma1 = np.array([1, 0.5])    # MA coefficient
# Generate a sample of 1000 data points
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=len(price_data))

# Plot the generated time series for ARMA(1,1)
plt.figure()
plt.plot(ARMA_1, label='ARMA(1, 1) Sample')
plt.title('Simulated ARMA(1, 1) Process')
plt.xlim([0, 200])  # Adjust limits as needed
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()

# Step 6: Display the autocorrelation and partial autocorrelation plots for ARMA(1,1)
plt.figure()
plt.subplot(2, 1, 1)
plot_acf(ARMA_1, lags=20, ax=plt.gca())
plt.title('ACF of ARMA(1, 1) Process')

plt.subplot(2, 1, 2)
plot_pacf(ARMA_1, lags=20, ax=plt.gca())
plt.title('PACF of ARMA(1, 1) Process')

plt.tight_layout()
plt.show()

# Step 7: Define an ARMA(2,2) process with coefficients ar2 and ma2
ar2 = np.array([1, -0.33, 0.5])  # Adjust AR coefficients as needed
ma2 = np.array([1, 0.9, 0.3])     # Adjust MA coefficients as needed
# Generate a sample of 10,000 data points
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=len(price_data) * 10)

# Plot the generated time series for ARMA(2,2)
plt.figure()
plt.plot(ARMA_2, label='ARMA(2, 2) Sample', color='orange')
plt.title('Simulated ARMA(2, 2) Process')
plt.xlim([0, 200])  # Adjust limits as needed
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()

# Step 8: Display the autocorrelation and partial autocorrelation plots for ARMA(2,2)
plt.figure()
plt.subplot(2, 1, 1)
plot_acf(ARMA_2, lags=20, ax=plt.gca())
plt.title('ACF of ARMA(2, 2) Process')

plt.subplot(2, 1, 2)
plot_pacf(ARMA_2, lags=20, ax=plt.gca())
plt.title('PACF of ARMA(2, 2) Process')

plt.tight_layout()
plt.show()

```

OUTPUT:
SIMULATED ARMA(1,1) PROCESS:


<img width="1083" height="760" alt="image" src="https://github.com/user-attachments/assets/a1af3dc5-7b1d-465c-9d75-03cad77ee3e4" />




Partial Autocorrelation

<img width="1021" height="530" alt="image" src="https://github.com/user-attachments/assets/239d363f-7f29-4425-96c4-ba9f20237a2c" />


Autocorrelation

<img width="1042" height="775" alt="image" src="https://github.com/user-attachments/assets/27404f3b-cd86-4d7e-bd18-cc311efe1138" />




SIMULATED ARMA(2,2) PROCESS:

<img width="1045" height="736" alt="image" src="https://github.com/user-attachments/assets/d9a4a3fc-21dc-40f7-8b66-3dc0899938ce" />


Partial Autocorrelation
<img width="1029" height="741" alt="image" src="https://github.com/user-attachments/assets/5f8d9e84-185d-4eaf-b7db-ccba79339ddb" />







RESULT:

 Thus, a python program is created to fir ARMA Model successfully.

