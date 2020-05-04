# -*- coding: utf-8 -*-

import time
from multiprocessing import Pool
import numpy as np

def sum_square(number):
    s = 0
    for i in range(number):
        s += i * i
    return s

def updater(*a):
    print('stuff')

if __name__ == '__main__':
    numbers = range(10000)
    someList = list(range(10))
    
    np.random.seed(0)
    
    for i in range(2):
        print(someList)
        np.random.shuffle(someList)
        
        start_time = time.time()
        p = Pool()
        result = p.map_async(sum_square, numbers)#, chunksize=100)
        
        p.close()
        
        while result.ready()==False:
            jobsLeft = p._cache[result._job]._number_left*p._cache[result._job]._chunksize
            print('complete: {:.2%}'.format((len(numbers)-jobsLeft)/len(numbers)))
            time.sleep(0.2)

        p.join()
    
        end_time = time.time() - start_time
        
        
    
        print(f"Processing {len(numbers)} numbers took {end_time} time using multiprocessing.")
        
        # print(result.get()[0:5])
