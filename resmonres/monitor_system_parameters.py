import gzip
import multiprocessing
import sched
import signal
import subprocess
import time
from datetime import datetime
from time import sleep
from resmon.resmon import chprio, SystemMonitor

# import matplotlib
# matplotlib.use('agg')

from matplotlib import pyplot as plt

def read_lines(file, mode ='b', encoding ='utf-8'):
    with gzip.open(file, mode='r'+mode) if file.endswith('.gz') else open(file,mode='r'+mode) as f:
        for line in f:
            if mode == 'b':
                yield line.decode(encoding).replace('\n','')
            elif mode == 't':
                yield line.replace('\n','')

def exec_command(command):
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {'stdout': p.stdout.readlines(),'stderr':p.stderr.readlines()}

def sigterm(signum, frame):
    raise KeyboardInterrupt()

def run_monitoring_fun(log_file='/tmp/log.csv',delay = 1):
    signal.signal(signal.SIGTERM, sigterm)

    try:
        chprio(-20)
        scheduler = sched.scheduler(time.time, time.sleep)
        sm = SystemMonitor(outfile_name=log_file)

        i = 1
        starttime = time.time()
        while True:
            scheduler.enterabs(
                time=starttime + i * delay, priority=2, action=SystemMonitor.poll_stat, argument=(sm,))
            scheduler.run()
            i += 1
    except KeyboardInterrupt:
        sm.close()

def plot_save(t,xx,file,title,sig_names=None):
    f = plt.figure(figsize=(16, 8), dpi=90)
    f.add_axes() # WTF !?
    axes = f.subplots()
    f.suptitle(title)

    if isinstance(xx[0],list):
        [plt.plot(t, x) for x in xx]
        plt.legend(sig_names)
    else:
        plt.plot(t, xx)
    axes.set_xlabel('seconds')
    axes.set_ylabel('%')
    f.savefig(file)


class MonitorSysParams(object):
    p = None

    def __init__(self,log_path='.'):
        self.log_path = log_path
        self.log_file = log_path+'/cpu-mem-log.csv'
        self.gpu_log_file = log_path+'/gpu-log.csv'

    p_gpu:subprocess.Popen=None
    def __enter__(self):
        self.p = multiprocessing.Process(target=run_monitoring_fun,kwargs={'log_file':self.log_file})
        r = exec_command('nvidia-smi')
        if len(r['stderr'])==0:
            print('found working nvidia-smi')
            self.p_gpu = subprocess.Popen('nvidia-smi '
                                          '--query-gpu=timestamp,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used'
                                          ' --format=csv -l 1 > %s'%self.gpu_log_file,shell=True)
        else:
            self.p_gpu = None
        self.p.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.p_gpu:
            # self.p_gpu.send_signal(signal.CTRL_C_EVENT)
            self.p_gpu.terminate()
            self.p_gpu.kill()
        self.p.terminate()
        sleep(1)
        g = read_lines(self.log_file)
        columns = [c.replace(' ','') for c in next(g).split(',')]
        lines = [line.split(',') for line in g if line.startswith('15')]
        t = [int(line[columns.index('Timestamp')])  for line in lines]
        t0=t[0]
        t = [tt-t0 for tt in t]
        cpu_cols = [c for c in columns if c.startswith('%CPU')]
        cpus = [[float(line[columns.index(cpu_col)]) for line in lines] for cpu_col in cpu_cols]
        ram = [float(line[columns.index('%MEM')]) for line in lines]

        plot_save(t, cpus, self.log_path + '/cpu.png','cpu-usage',cpu_cols)
        plot_save(t, ram, self.log_path + '/mem.png','memory-usage',cpu_cols)

        if self.p_gpu:
            lines = [l.split(',') for l in read_lines(self.gpu_log_file)]
            columns = [c.replace(' ','').replace('[','').replace(']','').replace('%','') for c in lines.pop(0)]
            print(columns)
            epoch_start = datetime.utcfromtimestamp(0)
            t = [(self.parse_time(l) - epoch_start).total_seconds() - t0 for l in lines]
            gpu_params_names = [c for c in columns if 'utilization' in c]
            gpu_params = [[float(l[columns.index(p)].replace(' ','').replace('%','')) for l in lines]
                          for p in gpu_params_names]
            if len(gpu_params)>0:
                plot_save(t,gpu_params, self.log_path+'/gpu_util.png','gpu-usage',gpu_params_names)

            gpu_params_names = [c for c in columns if 'MiB' in c]
            gpu_params = [[float(l[columns.index(p)].replace(' ','').replace('%','').replace('MiB','')) for l in lines]
                          for p in gpu_params_names]
            if len(gpu_params)>0:
                plot_save(t,gpu_params,self.log_path+'/gpu_mem.png','gpu-memory-usage',gpu_params_names)

    def parse_time(self, l):
        try:
            t = datetime.strptime(l[0], "%Y/%m/%d %H:%M:%S.%f")
        except Exception:
            t = datetime.utcfromtimestamp(0)
        return t

def benchmark_numpy():
    import numpy as np
    from time import time

    # Let's take the randomness out of random numbers (for reproducibility)
    np.random.seed(0)

    size = 4096
    A, B = np.random.random((size, size)), np.random.random((size, size))
    C, D = np.random.random((size * 128,)), np.random.random((size * 128,))
    E = np.random.random((int(size / 2), int(size / 4)))
    F = np.random.random((int(size / 2), int(size / 2)))
    F = np.dot(F, F.T)
    G = np.random.random((int(size / 2), int(size / 2)))

    # Matrix multiplication
    N = 20
    t = time()
    for i in range(N):
        np.dot(A, B)
    delta = time() - t
    print('Dotted two %dx%d matrices in %0.2f s.' % (size, size, delta / N))
    del A, B

    # Vector multiplication
    N = 5000
    t = time()
    for i in range(N):
        np.dot(C, D)
    delta = time() - t
    print('Dotted two vectors of length %d in %0.2f ms.' % (size * 128, 1e3 * delta / N))
    del C, D

    # Singular Value Decomposition (SVD)
    N = 3
    t = time()
    for i in range(N):
        np.linalg.svd(E, full_matrices=False)
    delta = time() - t
    print("SVD of a %dx%d matrix in %0.2f s." % (size / 2, size / 4, delta / N))
    del E

    # Cholesky Decomposition
    N = 3
    t = time()
    for i in range(N):
        np.linalg.cholesky(F)
    delta = time() - t
    print("Cholesky decomposition of a %dx%d matrix in %0.2f s." % (size / 2, size / 2, delta / N))

    # # Eigendecomposition
    # t = time()
    # for i in range(N):
    #     np.linalg.eig(G)
    # delta = time() - t
    # print("Eigendecomposition of a %dx%d matrix in %0.2f s." % (size / 2, size / 2, delta / N))

if __name__ == '__main__':
    with MonitorSysParams('.'):
        sleep(1)
        benchmark_numpy()
        sleep(2)