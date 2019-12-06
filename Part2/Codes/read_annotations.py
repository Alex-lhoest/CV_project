import pandas as pd

f = pd.read_csv("238453.txt", header=None)

for _, row in f.iterrows():
    coord1 = row[0]
    coord2 = row[1]
    x1, y1 = coord1.split()
    x2, y2 = coord2.split()

    with open("test.txt", 'a') as outfile:
        outfile.write(x1 + ", " + y1 + ", " + x2 + ", " + y2 + "\n")
    