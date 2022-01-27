from tqdm import tqdm

import time


for ctr, i in zip(
    range(100), tqdm(range(100), desc="Loading…", ascii=False, ncols=100)
):
    print(i)
    time.sleep(0.02)

print(ctr)
print("Complete.")
