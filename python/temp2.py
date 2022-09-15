# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 14:02:04 2022

@author: myf
"""
#简单测试1

import torch
print(torch.__version__)
from torch.utils import benchmark

typ = torch.float16
n = 1024*16
a = torch.randn(n,n).type(typ).cuda()
b = torch.randn(n,n).type(typ).cuda()

t = benchmark.Timer(
    stmt='a @ b',
    globals={'a':a,'b':b})

x = t.timeit(50)
print(2*n**3 /x.median / 1e12)
"""
import torch
import inspect
from collections import defaultdict
import pandas as pd
from torch.utils import benchmark 

pd.options.display.precision = 3

def var_dict(*args):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return dict([(name, val) for name, val in callers_local_vars if val is arg][0] 
                for arg in args)

def walltime(stmt, arg_dict, duration=3):
    return benchmark.Timer(stmt=stmt, globals=arg_dict).blocked_autorange(
        min_run_time=duration).median


matmul_tflops = defaultdict(lambda: {})
for n in [128, 512, 2048, 8192]:
    for dtype in (torch.float32, torch.float16):
        a = torch.randn(n, n, dtype=dtype).cuda()
        b = torch.randn(n, n, dtype=dtype).cuda()   
        t = walltime('a @ b', var_dict(a, b))
        matmul_tflops[f'n={n}'][dtype] = 2*n**3 / t / 1e12
        del a, b
        
pd.DataFrame(matmul_tflops)
"""