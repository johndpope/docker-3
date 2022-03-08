import os
from tkinter import S

filepath = r"G:\metric-is50k.txt"

with open(filepath, "r") as f:
    textfile = f.readlines()

textfile_new = []
for line in textfile:
    if not any([line.split(" ")[0] in line_new for line_new in textfile_new]):
        textfile_new.append(line)
textfile = sorted(textfile_new)

print(textfile)
with open(filepath, "w") as f:    
    f.writelines(sorted(textfile))