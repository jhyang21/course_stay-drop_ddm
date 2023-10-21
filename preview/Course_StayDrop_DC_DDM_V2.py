# This is Version 2 of Course Stay/Drop Drift-Changing Drift Diffusion Model

# Updates:
# well-connectedness is changed from a categorical variable to continuous variable  
# bias as a continuous variable is introduced

import numpy as np
import matplotlib.pyplot as plt
import statistics

# Function to simulate the modified drift-diffusion model
def simulate_ddm(T, dt, mu, sigma, event_probability, threshold):
    time_steps = int(T / dt)
    X = np.zeros(time_steps)
    decision_time = None
    
    # Introduce bias
    X[0] = np.random.normal(0, 0.20) + 0.25

    # Random event based on well-connectedness probability C
    for t in range(1, time_steps):
        if np.random.rand() < event_probability:  # Random event based on C
            if np.random.rand() < 0.80: # 80% chance of helpful
                mu += np.random.normal(0, 0.1)  # Change in mu
            else: # 20% chance of not helpful
                mu -= np.random.normal(0, 0.1)  # Change in mu
    
        # Drift term
        drift = mu

        # Noise term
        noise = sigma * np.sqrt(dt) * np.random.normal()

        # Update the decision variable
        X[t] = X[t-1] + drift * dt + noise

        # Check if the decision threshold is reached
        if abs(X[t]) >= threshold:
            decision_time = t * dt
            break

    if decision_time is None:
        decision_time = 21  # Set to 21 if decision_time is None

    return X, decision_time

# Simulation parameters
T = 21  # Total time in days
dt = 1/24  # Time step in a day (hours)
mu = 0.2  # Base drift rate
sigma = 0.1  # Noise level
threshold = 1.0  # Initial decision threshold
trials = 1000  # Number of trials for each probability

# Initialize plots
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_title('Drift Diffusion Model for Varying Well-connectedness Probabilities')

decision_times = []

# Run trials with varying well-connectedness probabilities
for i in range(trials):
    probability = np.random.normal(0.25,0.05)
    X, decision_time = simulate_ddm(T, dt, mu, sigma, probability, threshold)
    decision_times.append(decision_time)
    ax.plot(np.arange(0, T, dt)[:len(X)], X, alpha=0.5)


# Calculate and display results
avg_decision_time = np.mean(decision_times)
med_decision_time = statistics.median(decision_times)
percentage_over_7_days_below_14_days = len([t for t in decision_times if 7 < t <= 14]) / trials * 100
percentage_over_14_days = len([t for t in decision_times if t > 14]) / trials * 100

print(f"Mean Decision Time: {avg_decision_time:.2f} days")
print(f"Median Decision Time: {med_decision_time:.2f} days")
print(f"Percentage of trials taking longer than 7 days but below 14 days: {percentage_over_7_days_below_14_days:.2f}%")
print(f"Percentage of trials taking longer than 14 days: {percentage_over_14_days:.2f}%")


# Add labels, legend, and threshold lines
ax.set_xlabel('Time (days)')
ax.set_ylabel('Decision Variable')
ax.axhline(y= threshold, color='r', linestyle='--', label='Threshold')
ax.axhline(y=-threshold, color='r', linestyle='--')
ax.legend()

# Show the figure
plt.show()


