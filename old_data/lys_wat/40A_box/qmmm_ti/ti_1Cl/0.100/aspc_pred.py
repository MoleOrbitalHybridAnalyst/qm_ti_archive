import numpy as np
import time
import math

class SimpleQueue:
    def __init__(self, N):
        self.N = N
        self._data = list()

    def append(self, ele):
        if len(self._data) == self.N:
            self._data.pop(0)
        self._data.append(ele)
    
    def pop(self):
        return self._data.pop(0)

    def size(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data.__getitem__(item)

class DMPredictor:
    def __init__(self, n_diis=3):
        self.n_diis = n_diis
        assert self.n_diis > 0
        self.queue = SimpleQueue(n_diis)

    def append(self, ele):
        self.queue.append(ele)

    def predict(self):
        if self.queue.size() == 0:
            return None
        elif self.queue.size() < self.n_diis:
            return self.queue[0]
        else:
            t = time.time()
            nvec = self.queue.size() 
            print("---------------------------------------------")
            print(f"ASPC: predict using {nvec} points")
            pred = 0
            for i in range(self.n_diis):
                m = self.n_diis - i
                coeff = (-1)**(m+1) * m * math.comb(2*nvec, nvec-m) / math.comb(2*nvec-2, nvec-1)
                print(f" B({m}) = {coeff}")
                pred += coeff * self.queue[i]
            print(f"Wall Time: {time.time()-t} s")
            print("---------------------------------------------")
            return pred

