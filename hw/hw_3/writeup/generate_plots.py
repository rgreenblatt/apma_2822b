import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv("../benchmark_data.csv")

threads_16 = df.loc[df['threads'] == 16]

threads_16_size_2048 = threads_16.loc[threads_16['n'] == 2048]
threads_16_size_2048_blas = threads_16_size_2048.loc[threads_16_size_2048['used'] == 'blas']
threads_16_size_2048_mine = threads_16_size_2048.loc[threads_16_size_2048['used'] == 'mine']

threads_16_size_4096 = threads_16.loc[threads_16['n'] == 4096]
threads_16_size_4096_blas = threads_16_size_4096.loc[threads_16_size_4096['used'] == 'blas']
threads_16_size_4096_mine = threads_16_size_4096.loc[threads_16_size_4096['used'] == 'mine']

threads_16_size_8192 = threads_16.loc[threads_16['n'] == 8192]
threads_16_size_8192_blas = threads_16_size_8192.loc[threads_16_size_8192['used'] == 'blas']
threads_16_size_8192_mine = threads_16_size_8192.loc[threads_16_size_8192['used'] == 'mine']

threads_16_size_16384 = threads_16.loc[threads_16['n'] == 16384]
threads_16_size_16384_blas = threads_16_size_16384.loc[threads_16_size_16384['used'] == 'blas']
threads_16_size_16384_mine = threads_16_size_16384.loc[threads_16_size_16384['used'] == 'mine']

plt.plot(threads_16_size_2048_blas['nodes'].values, threads_16_size_2048_blas['gflops'].values, '--')
plt.plot(threads_16_size_4096_blas['nodes'].values, threads_16_size_4096_blas['gflops'].values, '--')
plt.plot(threads_16_size_8192_blas['nodes'].values, threads_16_size_8192_blas['gflops'].values, '--')
plt.plot(threads_16_size_16384_blas['nodes'].values, threads_16_size_16384_blas['gflops'].values, '--')

plt.plot(threads_16_size_2048_mine['nodes'].values, threads_16_size_2048_mine['gflops'].values)
plt.plot(threads_16_size_4096_mine['nodes'].values, threads_16_size_4096_mine['gflops'].values)
plt.plot(threads_16_size_8192_mine['nodes'].values, threads_16_size_8192_mine['gflops'].values)
plt.plot(threads_16_size_16384_mine['nodes'].values, threads_16_size_16384_mine['gflops'].values)

plt.legend(['size 2048, blas',
            'size 4096, blas',
            'size 8192, blas',
            'size 16384, blas',
            'size 2048, mine',
            'size 4096, mine',
            'size 8192, mine',
            'size 16384, mine'])

plt.title('GFLOPS vs nodes for 16 threads')
plt.xlabel('nodes')
plt.ylabel('GFLOPS')

plt.savefig('gflops_nodes_16_threads')
plt.clf()

get_total_time = lambda df: np.array(df['setup']) + np.array(df['wait']) + np.array(df['copy'])

plt.plot(threads_16_size_2048_blas['nodes'].values, get_total_time(threads_16_size_2048_blas), '--')
plt.plot(threads_16_size_4096_blas['nodes'].values, get_total_time(threads_16_size_4096_blas), '--')
plt.plot(threads_16_size_8192_blas['nodes'].values, get_total_time(threads_16_size_8192_blas), '--')
plt.plot(threads_16_size_16384_blas['nodes'].values, get_total_time(threads_16_size_16384_blas), '--')

plt.plot(threads_16_size_2048_mine['nodes'].values, get_total_time(threads_16_size_2048_mine))
plt.plot(threads_16_size_4096_mine['nodes'].values, get_total_time(threads_16_size_4096_mine))
plt.plot(threads_16_size_8192_mine['nodes'].values, get_total_time(threads_16_size_8192_mine))
plt.plot(threads_16_size_16384_mine['nodes'].values, get_total_time(threads_16_size_16384_mine))

plt.legend(['size 2048, blas',
            'size 4096, blas',
            'size 8192, blas',
            'size 16384, blas',
            'size 2048, mine',
            'size 4096, mine',
            'size 8192, mine',
            'size 16384, mine'])

plt.title('Time spent moving data between processes for 16 threads')
plt.xlabel('nodes')
plt.ylabel('time (s)')

plt.savefig('time_nodes_16_threads')
plt.clf()

thread_1 = df.loc[df['threads'] == 1]

thread_1_size_2048 = thread_1.loc[thread_1['n'] == 2048]
thread_1_size_2048_blas = thread_1_size_2048.loc[thread_1_size_2048['used'] == 'blas']
thread_1_size_2048_mine = thread_1_size_2048.loc[thread_1_size_2048['used'] == 'mine']

thread_1_size_4096 = thread_1.loc[thread_1['n'] == 4096]
thread_1_size_4096_blas = thread_1_size_4096.loc[thread_1_size_4096['used'] == 'blas']
thread_1_size_4096_mine = thread_1_size_4096.loc[thread_1_size_4096['used'] == 'mine']

thread_1_size_8192 = thread_1.loc[thread_1['n'] == 8192]
thread_1_size_8192_blas = thread_1_size_8192.loc[thread_1_size_8192['used'] == 'blas']
thread_1_size_8192_mine = thread_1_size_8192.loc[thread_1_size_8192['used'] == 'mine']

thread_1_size_16384 = thread_1.loc[thread_1['n'] == 16384]
thread_1_size_16384_blas = thread_1_size_16384.loc[thread_1_size_16384['used'] == 'blas']
thread_1_size_16384_mine = thread_1_size_16384.loc[thread_1_size_16384['used'] == 'mine']

plt.plot(thread_1_size_2048_blas['processes'].values, thread_1_size_2048_blas['gflops'].values, '--')
plt.plot(thread_1_size_4096_blas['processes'].values, thread_1_size_4096_blas['gflops'].values, '--')
plt.plot(thread_1_size_8192_blas['processes'].values, thread_1_size_8192_blas['gflops'].values, '--')
plt.plot(thread_1_size_16384_blas['processes'].values, thread_1_size_16384_blas['gflops'].values, '--')

plt.plot(thread_1_size_2048_mine['processes'].values, thread_1_size_2048_mine['gflops'].values)
plt.plot(thread_1_size_4096_mine['processes'].values, thread_1_size_4096_mine['gflops'].values)
plt.plot(thread_1_size_8192_mine['processes'].values, thread_1_size_8192_mine['gflops'].values)
plt.plot(thread_1_size_16384_mine['processes'].values, thread_1_size_16384_mine['gflops'].values)

plt.legend(['size 2048, blas',
            'size 4096, blas',
            'size 8192, blas',
            'size 16384, blas',
            'size 2048, mine',
            'size 4096, mine',
            'size 8192, mine',
            'size 16384, mine'])

plt.title('GFLOPS vs processes for 1 thread')
plt.xlabel('processes')
plt.ylabel('GFLOPS')

plt.savefig('gflops_processes_1_thread')
plt.clf()

plt.plot(thread_1_size_2048_blas['processes'].values, get_total_time(thread_1_size_2048_blas), '--')
plt.plot(thread_1_size_4096_blas['processes'].values, get_total_time(thread_1_size_4096_blas), '--')
plt.plot(thread_1_size_8192_blas['processes'].values, get_total_time(thread_1_size_8192_blas), '--')
plt.plot(thread_1_size_16384_blas['processes'].values, get_total_time(thread_1_size_16384_blas), '--')

plt.plot(thread_1_size_2048_mine['processes'].values, get_total_time(thread_1_size_2048_mine))
plt.plot(thread_1_size_4096_mine['processes'].values, get_total_time(thread_1_size_4096_mine))
plt.plot(thread_1_size_8192_mine['processes'].values, get_total_time(thread_1_size_8192_mine))
plt.plot(thread_1_size_16384_mine['processes'].values, get_total_time(thread_1_size_16384_mine))

plt.legend(['size 2048, blas',
            'size 4096, blas',
            'size 8192, blas',
            'size 16384, blas',
            'size 2048, mine',
            'size 4096, mine',
            'size 8192, mine',
            'size 16384, mine'])

plt.title('Time spent moving data between processes for 1 thread')
plt.xlabel('nodes')
plt.ylabel('time (s)')

plt.savefig('time_nodes_1_threads')
plt.clf()

print("mine max:", max(df.loc[df['used'] == 'mine']['gflops'].values))
print("mine blas:", max(df.loc[df['used'] == 'blas']['gflops'].values))

weak_scaling_mine = [4.19393, 4.27199, 5.49029]

weak_scaling_blas = [1.10788, 3.3129, 2.24377]

weak_scaling_cores = [1, 8, 64]

plt.plot(weak_scaling_cores, weak_scaling_mine)
plt.plot(weak_scaling_cores, weak_scaling_blas)

plt.title('Weak Scaling')
plt.xlabel('number of cores')
plt.ylabel('time (s)')

plt.legend(['mine',
            'blas'])

plt.savefig('weak_scaling')
plt.clf()
