import subprocess
import time
import re
import numpy as np
import os

def get_gpu_occ(t=2, N=10):
    usage = list()
    for i in range(N):
        output = str(subprocess.check_output(['nvidia-smi'])).split("\\n")
        for line in output:
            m = re.search("([0-9]+)%\s+Default", line)
            if m is not None:
                usage.append(int(m.group(1)))
        time.sleep(2)
    usage = np.array(usage).reshape(N, 5)
    return np.mean(usage[:,[0,1,2,4]], axis=0)

def next_lambda(l):
    if type(l) == str:
        l = float(l)
    if l >= 1:
        return None
    else:
        return "%.3f"%(float(l) + 0.05)

def prev_lambda(l):
    if type(l) == str:
        l = float(l)
    if l <= 0:
        return None
    else:
        return "%.3f"%(float(l) - 0.05)

def is_longer(fname, nlines0):
    with open(fname) as fp:
        nlines = len(fp.readlines())
        return nlines > nlines0

def prepare_and_run(source, target, gpuid):
    if os.path.exists(target):
        print(f"{target} exists!")
        exit(1)
    os.system(f"cp -rp template {target}")
    os.chdir(target)
    os.system(f"sh set_socket.sh lysqm{target} lysmm{target}")
    with open("run.sh") as fp:
        contents = fp.readlines()
    with open("run.sh", "w") as fp:
        for line in contents:
            line = re.sub("YYY", str(gpuid), line)
            line = re.sub("ZZZ", target, line)
            fp.write(line)
    os.system(f"bash ../new_geom.sh {source}")
    os.system("nohup bash run.sh &")
    os.chdir("../")

def main():
    l0 = 0
    while l0 is not None:
        l0 = next_lambda(l0)
        if not os.path.isfile(l0 + "/simulation.out"):
            break
    l0 = prev_lambda(l0)

    l1 = 1
    while l1 is not None:
        l1 = prev_lambda(l1)
        if not os.path.isfile(l1 + "/simulation.out"):
            break
    l1 = next_lambda(l1)

    while True:
        os.system("date")
        print("Current l0, l1 =", l0, l1)
        gpu_occ = get_gpu_occ(t=3, N=40)
        print("GPU occupancy:", *gpu_occ)
        idlest_gpu = np.argmin(gpu_occ)
        print("The idlest GPU is:", idlest_gpu)
        if gpu_occ[idlest_gpu] < 40:
            print("Its occupancy < 40%")
            idle_lv = 2
        elif gpu_occ[idlest_gpu] >= 40 and gpu_occ[idlest_gpu] < 70:
            print("Its occupancy > 40% but < 70%")
            idle_lv = 1
        else:
            print("Its occupancy > 70%")
            idle_lv = 0
            continue
        if idle_lv == 2:
            if is_longer(f"{l1}/simulation.out", 3011):
                nextl = prev_lambda(l1)
                print(f"Prepare {nextl} from {l1} and run on GPU {idlest_gpu}")
                prepare_and_run(l1, nextl, idlest_gpu)
                l1 = nextl
            elif is_longer(f"{l0}/simulation.out", 3011):
                nextl = next_lambda(l0)
                print(f"Prepare {nextl} from {l0} and run on GPU {idlest_gpu}")
                prepare_and_run(l0, nextl, idlest_gpu)
                l0 = nextl
        elif idle_lv == 1:
            if is_longer(f"{l1}/simulation.out", 5111):
                nextl = prev_lambda(l1)
                print(f"Prepare {nextl} from {l1} and run on GPU {idlest_gpu}")
                prepare_and_run(l1, nextl, idlest_gpu)
                l1 = nextl
            elif is_longer(f"{l0}/simulation.out", 5111):
                nextl = next_lambda(l0)
                print(f"Prepare {nextl} from {l0} and run on GPU {idlest_gpu}")
                prepare_and_run(l0, nextl, idlest_gpu)
                l0 = nextl
        print("Sleep")
        print("")
        time.sleep(600)

if __name__ == "__main__":
    main()
