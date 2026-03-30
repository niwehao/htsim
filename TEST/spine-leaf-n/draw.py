import os
import re
import matplotlib.pyplot as plt
import numpy as np

times = []
for i in range(1, 101):
    filepath = f"log/{i}.txt"
    if not os.path.exists(filepath):
        continue
    with open(filepath, "r") as f:
        lines = f.readlines()
        if len(lines) >= 2:
            match = re.search(r"Simulation ended at t=([\d.]+)\s*ms", lines[-1])
            if match:
                #print(f"File: {filepath}, Simulation Time: {match.group(1)} ms")
                times.append(float(match.group(1)))

times.sort()
cdf = np.arange(1, len(times) + 1) / len(times)

plt.figure(figsize=(8, 5))
plt.step(times, cdf, where="post")
plt.xlabel("Simulation Time (ms)")
plt.ylabel("CDF")
plt.title("CDF of Simulation End Times")
plt.grid(True)
plt.tight_layout()
plt.savefig("cdf.png", dpi=150)
plt.show()