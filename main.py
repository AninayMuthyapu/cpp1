import pytorch
import torch
m = 8*1024
n = 8*1024
k = 8*1024
a = torch.randn((m,k),dtype=torch.float32)
b = torch.randn((k,n),dtype=torch.float32)

import timeit, time
torch.set_num_threads(64)
c = a@b
c = a@b
start = time.time()
for i in range(20):
    c = a@b
    end = time.time()
    t = (end-start)/20
    print ("Time is : ", t * 1000, " ms ")
    print ("GFLOPS: ", (2*a.shape[0]*a.shape[1]*b.shape[1])/t/1e9)
 
  