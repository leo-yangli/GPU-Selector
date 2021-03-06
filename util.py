import os
import numpy as np
import time

def get_available_gpu(num_gpu=1, min_memory=1000, sample=3, nitro_restriction=True, verbose=True):
    '''
    :param num_gpu: number of GPU you want to use
    :param min_memory: minimum memory
    :param sample: number of sample
    :param nitro_restriction: if True then will not distribute the last GPU for you.
    :param verbose: verbose mode
    :return: str of best choices, e.x. '1, 2'
    '''
    sum = None
    for _ in range(sample):
        info = os.popen('nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv').read()
        info = np.array([[id] + t.replace('%', '').replace('MiB','').split(',') for id, t in enumerate(info.split('\n')[1:-1])]).\
            astype(np.int)
        sum = info + (sum if sum is not None else 0)
        time.sleep(0.2)
    avg = sum//sample
    if nitro_restriction:
        avg = avg[:-1]
    available = avg[np.where(avg[:,2] > min_memory)]    
    if available.shape[0] == 0:
        print('No GPU available')
        return ''
    select = ', '.join(available[np.argsort(available[:,1])[:num_gpu],0].astype(np.str).tolist())
    if verbose:
        print('Available GPU List')
        first_line = [['id', 'utilization.gpu(%)', 'memory.free(MiB)']]
        matrix = first_line + available.astype(np.int).tolist()
        s = [[str(e) for e in row] for row in matrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))
        print('Select id #' + select + ' for you.')
    return select
