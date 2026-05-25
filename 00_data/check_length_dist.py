import jsonlines
import sys
import numpy as np
from collections import defaultdict

cois = ["prompt", "reasoning_language", "answer"]
path = sys.argv[1]
print(path)
d = defaultdict(list)
N = 0
total_None_count = 0
with jsonlines.open(path, 'r') as f:
    for line_no, obj in enumerate(f):
        for c in cois:
            if obj[c] is not None:
                d[c] += [len(obj[c].split())]
            else:
                total_None_count += 1
                print("None value encountered in line", line_no)
        N += 1

d_mean = {k+"_len_mean": np.mean(v) for (k,v) in d.items()}
d_std = {k+"_len_std": np.std(v, ddof=1) for (k,v) in d.items()}
d_final = {**d_mean, **d_std}

total_len_mean = sum(d_final[c+"_len_mean"] for c in cois)
total_len_std = sum(d_final[c+"_len_std"] for c in cois)

print(d_final)
print("N examples", N)
print(f"Mean total len {cois}: {total_len_mean}")
print(f"STD total len {cois}: {total_len_std}")
print(f"Total None count: {total_None_count}")

