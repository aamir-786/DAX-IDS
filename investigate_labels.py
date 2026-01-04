import numpy as np
import matplotlib.pyplot as plt

y = np.load('CIC-IDS2017/y.npy')
y_subset = y[:500000]

print(f"Total Positives: {np.sum(y_subset)}")

# Plot running mean of labels to see if they are clustered
def running_mean(x, N):
    return np.convolve(x, np.ones(N)/N, mode='valid')

rm = running_mean(y_subset, 5000)
plt.figure(figsize=(10, 5))
plt.plot(rm)
plt.title("Label distribution (Running Mean, window=5000)")
plt.xlabel("Sample Index")
plt.ylabel("Positive Ratio")
plt.savefig("label_distribution_cic.png")

n = 500000
te = int(0.6 * n)
ve = int(0.8 * n)

y_train = y_subset[:te]
y_val = y_subset[te:ve]
y_test = y_subset[ve:]

print(f"Train Positive Ratio: {np.sum(y_train)/len(y_train):.4%}")
print(f"Val Positive Ratio: {np.sum(y_val)/len(y_val):.4%}")
print(f"Test Positive Ratio: {np.sum(y_test)/len(y_test):.4%}")
