"""Thread-safe SDK byte-storage double with guarded registered destinations.

FlexKV batching, credits, result checks and worker processing stay real.
This implements only the Mooncake SDK endpoint, without RDMA or a service.
"""
import ctypes
import threading
import time
import numpy as np


class ByteStore:
    def __init__(self):
        self.objects={};self.lock=threading.Lock();self.ranges=[]
        self.calls=[];self.active=0;self.peak=0;self.active_bytes=0;self.peak_bytes=0
        self.get_hook=None;self.put_hook=None;self.delay=0
    def register(self, array):
        self.ranges.append((array.ctypes.data,array.ctypes.data+array.nbytes))
    def check(self, ptr, size):
        assert any(lo<=ptr and ptr+size<=hi for lo,hi in self.ranges), 'unregistered/out-of-range DMA'
    def batch_is_exist(self, keys):
        with self.lock:return [int(k in self.objects) for k in keys]
    def batch_put_from(self, keys, ptrs, sizes):
        results=[]
        for i,(key,ptr,size) in enumerate(zip(keys,ptrs,sizes)):
            self.check(ptr,size)
            status=self.put_hook(key,i) if self.put_hook else 0
            if status==0:
                value=ctypes.string_at(ptr,size)
                with self.lock:self.objects[key]=value
            results.append(status)
        with self.lock:self.calls.append(('put',list(keys),list(sizes)))
        return results
    def batch_get_into(self, keys, ptrs, sizes):
        cost=sum(((n+4095)//4096)*4096+8192 for n in sizes)
        with self.lock:
            self.active+=1;self.active_bytes+=cost
            self.peak=max(self.peak,self.active);self.peak_bytes=max(self.peak_bytes,self.active_bytes)
            self.calls.append(('get',list(keys),list(sizes)))
        try:
            if self.delay:time.sleep(self.delay*(1+sum(map(len,keys))%3))
            results=[]
            for i,(key,ptr,size) in enumerate(zip(keys,ptrs,sizes)):
                self.check(ptr,size)
                with self.lock:value=self.objects.get(key)
                status=self.get_hook(key,i,size) if self.get_hook else (len(value) if value is not None else -1)
                if value is not None and status>0:ctypes.memmove(ptr,value,min(status,size,len(value)))
                results.append(status)
            return results
        finally:
            with self.lock:self.active-=1;self.active_bytes-=cost


def guarded_buffers(store, sizes, seed=1):
    rng=np.random.default_rng(seed);arrays=[];pointers=[];values=[]
    for size in sizes:
        array=np.full(size+66,0xA5,dtype=np.uint8)
        values.append(rng.integers(0,256,size,dtype=np.uint8).tobytes())
        arrays.append(array);pointers.append(array.ctypes.data+33);store.register(array)
    return arrays,pointers,values


def assert_guards(arrays):
    for array in arrays:
        assert np.all(array[:33]==0xA5) and np.all(array[-33:]==0xA5), 'DMA crossed an object boundary'
