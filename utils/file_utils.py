import numpy as np
import shutil
import os
import logging
from scipy.io import wavfile
import pickle, traceback

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm

# A logger for this file
log = logging.getLogger(__name__)

def copyFile(inFname, outFname):
    shutil.copyfile(inFname, outFname)
    log.info("copy file {} to {}".format(inFname, outFname))

def loadPkl(filename):
    try:
        file = open(filename, 'rb')
        data = pickle.load(file, encoding='latin1')
        file.close()
        return data
    except pickle.UnpicklingError as e:
        # normal, somewhat expected
        traceback.print_exc()
    except (AttributeError,  EOFError, ImportError, IndexError) as e:
        # secondary errors
        traceback.print_exc()
    except Exception as e:
        # everything else, possibly fatal
        traceback.print_exc()
        return

def dumpPkl(data, filename):
    try:
        file = open(filename, 'wb')
        data = pickle.dump(data, file)
        file.close()
        log.info("saved in {}".format(filename))
    except Exception as e:
        # everything else, possibly fatal
        traceback.print_exc()
        return

def get_basename_wo_extension(path):
    return os.path.basename(path).split('.')[0]


def _read_wav(path):
    fs, data = wavfile.read(path)
    return fs, data

def read_wav(paths):
    """Return: list of (fs, data, min(data), max(data))"""
    return run_parallel(task=_read_wav, args_list=[[p] for p in paths])

def _run_parallel_chunk(task, args_list, idx):
    # create a thread pool
    with ThreadPoolExecutor(len(args_list)) as exe:
        futures = [exe.submit(task, *args) for args in args_list]
        results = [future.result() for future in futures]
        return results, idx

def run_parallel(task, args_list):
    # n_worker = n_process = n_cpu, chunk_size is the n_thread per process
    n_worker = os.cpu_count()
    n_task = len(args_list)
    chunksize = max(round(n_task / n_worker), 1)
    # log.info("{}() * {} = n_worker {} * chunksize {}".format(task.__name__, n_task, n_worker, chunksize))
    
    # create the process pool
    pbar = tqdm(total=n_task, desc="Loading dataset")
    with ProcessPoolExecutor(n_worker) as executor:
        # split the load operations into chunks
        args_list_chunks = [args_list[i:(i + chunksize)] for i in range(0, n_task, chunksize)]
        futures = [executor.submit(_run_parallel_chunk, task=task, args_list=a, idx=i) for i, a in enumerate(args_list_chunks)]
        # process all results
        res_list = [()]*n_task # maintain the same order with args_list
        for future in as_completed(futures):
            results, chunk_idx = future.result() # results per chunk
            pbar.update(n=len(results))
            for i in range(len(results)):
                res_list[chunk_idx*chunksize+i] = results[i]
        return res_list

