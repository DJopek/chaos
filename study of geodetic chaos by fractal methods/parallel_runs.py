import subprocess
processes = []
for i in range(5):
    p = subprocess.Popen(["python", "../trajectories.py", str(i+1)])
    processes.append(p)
for p in processes:
    p.wait()