# -*- coding: utf-8 -*-

import time
import multiprocessing
import numpy as np

def sumSq(number):
    s = 0
    for i in range(number):
        s += i * i
    return s

class Consumer(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        # proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                self.task_queue.task_done()
                break
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return

class Task(object):
    def __init__(self, a):
        self.a = a
    def __call__(self):
        s = 0
        for i in range(self.a):
            s += i * i
        return s
    
if __name__ == '__main__':
    import backpropagation as bp
    import mnist
    
    # numbers = range(5000)
    someList = list(range(50000))
    
    t=bp.trainer()
    t.initialiseNetwork([784,16,16,10], (0,))
    
    data = mnist.database(True)
    
    np.random.seed(0)    
    
    for i in range(1):
        np.random.shuffle(someList)
        
        numbers = someList[0:100]

        start_time = time.time()
        for j in numbers:
            sumSq(j)
        print('serial execution time: {:n}s.'.format(time.time()-start_time))
        
        start_time = time.time()
        
        # Establish communication queues
        tasks = multiprocessing.JoinableQueue()
        results = multiprocessing.Queue()
        
        # Start consumers
        num_consumers = multiprocessing.cpu_count()
        consumers = [ Consumer(tasks, results)
                      for i in range(num_consumers) ]
        for w in consumers:
            w.start()

        print('consumers started: {:n}'.format(time.time()-start_time))
        
        # Enqueue jobs
        num_jobs = len(numbers)
        for i in numbers:
            tasks.put(Task(i))
        
        # Add a poison pill for each consumer
        for i in range(num_consumers):
            tasks.put(None)
    
        print('jobs all started: {:n}'.format(time.time()-start_time))
        
        while tasks.qsize() > 0:
            print('{:0.1%}'.format(1-tasks.qsize()/(len(numbers)+num_consumers)))
            time.sleep(0.5)

        # Wait for all of the tasks to finish
        tasks.join()

        end_time = time.time() - start_time
        
        print('parallel execution time (inc overheads): {:n}s'.format(end_time))

        while num_jobs:
            result = results.get()
            # print ('Result:', result)
            num_jobs -= 1