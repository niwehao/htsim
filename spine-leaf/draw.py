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
        # 从后往前找最后一个 INFERENCE FINISHED (DRAGONFLY)
        for line in reversed(lines):
            match = re.search(r"\[Node \d+\] INFERENCE FINISHED\s+t=([\d.]+)\s*ms", line)
            if match:
                times.append(float(match.group(1))*8)
                break

times.sort()
cdf = np.arange(1, len(times) + 1) / len(times)

plt.figure(figsize=(8, 5))
plt.step(times, cdf, where="post", label="AYNC")
plt.axvline(x=46.40*16, color="red", linestyle="--", label="SYNC")
plt.xlabel("Simulation Time (ms)")
plt.ylabel("CDF")
plt.title("Spine-leaf:OGPUs=128 Leaves=16 Spines=8 PortsPerSwitch=16 \n Layers=32 Frags=512x4KB Link=200Gbps ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cdf.png", dpi=150)
plt.show()