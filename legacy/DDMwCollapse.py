import numpy as np
import matplotlib.pyplot as plt
import statistics

# Function to simulate the modified drift-diffusion model
def simulate_ddm(T, dt, mu, sigma, C, initial_threshold, collapse_rate):
    time_steps = int(T / dt)
    X = np.zeros(time_steps)
    decision_time = None
    threshold = initial_threshold

    for t in range(1, time_steps):
        threshold = max(0.1, initial_threshold - collapse_rate * t * dt)
        if C == 1:
            if np.random.rand() < 0.10:  # 10% chance of a random event
                if np.random.rand() < 0.80: # 80% chance of helpful
                    mu += np.random.normal(0, 0.1)  # Change in mu
                else: # 20% chance of not helpful
                    mu -= np.random.normal(0, 0.1)  # Change in mu
        else:
            if np.random.rand() < 0.05:  # 5% chance of a random event
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
        decision_time = 21  # Set to 14 if decision_time is None

    return X, decision_time

# Simulation parameters
T = 21  # Total time in days
dt = 1/24  # Time step in a day (hours)
mu = 0.2  # Base drift rate
sigma = 0.1  # Noise level
initial_threshold = 1.0  # Initial decision threshold
collapse_rate = 0.02 #Collapse rate
trials = 1000  # Number of trials

# Initialize plots and decision time accumulators
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].set_title('Well-connected Student')
axes[1].set_title('Not Well-connected Student')
decision_times1 = []
decision_times2 = []
count_over_7_days_below_14_days1 = 0
count_over_7_days_below_14_days2 = 0
count_over_14_days1 = 0
count_over_14_days2 = 0

# Run 100 trials for well-connected student
for i in range(trials):
    C = 1  # Well-connected student (1 for yes, 0 for no)
    X1, decision_time1 = simulate_ddm(T, dt, mu, sigma, C, initial_threshold, collapse_rate)
    decision_times1.append(decision_time1)
    if decision_time1 > 14:
        count_over_14_days1 += 1
    elif decision_time1 > 7:
        count_over_7_days_below_14_days1 += 1
    axes[0].plot(np.arange(0, T, dt)[:len(X1)], X1, alpha=0.5)

# Run 100 trials for not well-connected student
for i in range(trials):
    C = 0  # Not well-connected student (1 for yes, 0 for no)
    X2, decision_time2 = simulate_ddm(T, dt, mu, sigma, C, initial_threshold, collapse_rate)
    decision_times2.append(decision_time2)
    if decision_time2 > 14:
        count_over_14_days2 += 1
    elif decision_time2 > 7:
        count_over_7_days_below_14_days2 += 1
    axes[1].plot(np.arange(0, T, dt)[:len(X2)], X2, alpha=0.5)

# Calculate and display average decision times
avg_decision_time1 = np.mean(decision_times1)
avg_decision_time2 = np.mean(decision_times2)

# Calculate and display median decision times
med_decision_time1 = statistics.median(decision_times1)
med_decision_time2 = statistics.median(decision_times2)

# Calculate percentage of trials taking longer than 7 days below 14 days
percentage_over_7_days_below_14_days1 = (count_over_7_days_below_14_days1 / trials) * 100
percentage_over_7_days_below_14_days2 = (count_over_7_days_below_14_days2 / trials) * 100


# Calculate percentage of trials taking longer than 14 days
percentage_over_14_days1 = (count_over_14_days1 / trials) * 100
percentage_over_14_days2 = (count_over_14_days2 / trials) * 100

# Show results
print(f"Average Decision Time for Well-connected Student: {avg_decision_time1:.2f} days")
print(f"Average Decision Time for Not Well-connected Student: {avg_decision_time2:.2f} days")
print(f"Median Decision Time for Well-connected Student: {med_decision_time1:.2f} days")
print(f"Median Decision Time for Not Well-connected Student: {med_decision_time2:.2f} days")
print(f"Percentage of trials taking longer than 7 days below 14 days for well-connected student: {percentage_over_7_days_below_14_days1:.2f}%")
print(f"Percentage of trials taking longer than 7 days below 14 days for not well-connected student: {percentage_over_7_days_below_14_days2:.2f}%")
print(f"Percentage of trials taking longer than 14 days for well-connected student: {percentage_over_14_days1:.2f}%")
print(f"Percentage of trials taking longer than 14 days for not well-connected student: {percentage_over_14_days2:.2f}%")

# Add threshold lines and labels
for ax in axes:
    ax.axhline(y= initial_threshold, color='r', linestyle='--', label='Threshold')
    ax.axhline(y=-initial_threshold, color='r', linestyle='--')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Decision Variable')
    ax.legend()

plt.show()
