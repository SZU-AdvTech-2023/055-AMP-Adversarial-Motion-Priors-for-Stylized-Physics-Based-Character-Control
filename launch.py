# Fix pycharm bug when running subdirectory script

"""
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --network=host --gpus=all --name=<NAME> <ID> /bin/bash

# maps:
    ./IsaacGymEnvs -> /opt/projects

# args:
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --network=host --gpus=all
"""
import os
import time
import datetime
import sys


class PrintToConsole:
    def __init__(self, *args):
        pass

    def __enter__(self):
        self.old_stdout = sys.stdout
        sys.stdout = sys.__stdout__

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout

class PrintToLog:
    def __init__(self, filename: str):
        self.filename = filename

    def __enter__(self):
        self.log_file = open(self.filename, 'w')
        sys.stdout = self.log_file

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = sys.__stdout__
        self.log_file.close()

class Tick:
    def __init__(self, heading=' >>> '):
        self.heading = heading

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        delta_time = time.time() - self.start_time
        hours, remainder = divmod(int(delta_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"
        print(f"{self.heading}{formatted_time}")


def cleanup_failed_runs():
    pj = os.path.join
    for d in os.listdir("runs"):
        if not os.path.isdir(pj("runs", d)):
            continue
        p = pj("runs", pj(d, "nn"))
        if len(os.listdir(p)) == 0:
            import shutil
            print(f"cleaning {pj('runs', d)} ... ")
            shutil.rmtree(pj('runs', d))


def main():
    cwd = os.getcwd()
    print(f"cwd: {cwd}")

    cur_time = datetime.datetime.now()
    os.makedirs("./runs", exist_ok=True)
    filename = os.path.join("./runs",
                            f"{cur_time.hour:02d}-{cur_time.minute:02d}--"
                            f"{cur_time.month:02d}-{cur_time.day:02d}.log")
    cleanup_failed_runs()

    with PrintToConsole(filename):
        import isaacgym
        import torch
        print(' ################### ')
        print(f" cwd: {cwd}")
        print(f" torch: {torch.__version__}")
        print(' ################### ')

        from isaacgymenvs.train import launch_rlg_hydra
        with Tick("\n\n\n\n    run launch_rlg_hydra time elapsed: "):
            launch_rlg_hydra()


if __name__ == '__main__':
    main()
