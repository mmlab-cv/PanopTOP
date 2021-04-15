import os

for i in range(10, 29401, 10):
    file = f'trans/{i:08}.ply'
    print(i)

    if not os.path.exists(file):
        print("skip ", file)
        continue

    n = 11
    nfirstlines = []

    with open(file) as f, open(f"{file}tmp", "w") as out:
        for x in range(n):
            nfirstlines.append(next(f))
        for line in f:
            out.write(line)

    os.remove(file)
    os.rename(f"{file}tmp", file)
