"""Actual C++ hash.cpp + xxhash and Python SequenceMeta; no torch/CUDA.

The Tensor ABI shim supplies CPU pointers only. Hash algorithms and adapters
are repository source, while Mooncake SDK I/O is a byte-exact in-memory double.
"""
import ctypes
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import numpy as np
from submission_test_support import ROOT, NS, base, load_nodes
from mooncake_test_support import MooncakeStoreClient

spec=importlib.util.spec_from_file_location('hy4_key_codec',ROOT.parent/'sglang-v0.5.10-kunlun-main-hy4/python/sglang/srt/mem_cache/storage/flexkv/flexkv_key_codec.py')
codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec)

class NativeHashTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='hy4-native-hash-')
        tmp=Path(cls.tmp.name);(tmp/'torch').mkdir()
        (tmp/'omp.h').write_text('')
        (tmp/'torch/extension.h').write_text('''#pragma once
#include <cassert>
#include <cstddef>
namespace torch { class Tensor { void* ptr_; size_t n_, elem_; public:
Tensor(void* p,size_t n,size_t e):ptr_(p),n_(n),elem_(e){}
int dim() const{return 1;} size_t numel()const{return n_;}
size_t element_size()const{return elem_;} void* data_ptr()const{return ptr_;}
template<class T> T* data_ptr()const{return static_cast<T*>(ptr_);} }; }
''')
        # Remove binding-only declarations; retain the actual Hasher declaration.
        header=(ROOT/'csrc/cache_utils.h').read_text()
        header=header.replace('#include <pybind11/pybind11.h>','').replace('namespace py = pybind11;','')
        (tmp/'cache_utils.h').write_text(header)
        shutil.copyfile(ROOT/'csrc/hash.cpp',tmp/'hash.cpp')
        (tmp/'bridge.cpp').write_text('''#include "cache_utils.h"
extern "C" {
void* hnew(){return new flexkv::Hasher();}
void hfree(void* h){delete static_cast<flexkv::Hasher*>(h);}
void hreset(void* h){static_cast<flexkv::Hasher*>(h)->reset();}
void hupdate(void* h,void* p,size_t n){static_cast<flexkv::Hasher*>(h)->update(p,n);}
unsigned long long hdigest(void* h){return static_cast<flexkv::Hasher*>(h)->digest();}
void hgen(void* h,void* ids,size_t n,int page,void* out){
 torch::Tensor in(ids,n,8), result(out,n/page,8);
 flexkv::gen_hashes(*static_cast<flexkv::Hasher*>(h),in,page,result);}
}
''')
        lib=tmp/'hash.dylib'
        subprocess.run(['c++','-std=c++17','-shared','-fPIC','-O2','-I'+str(tmp),
                        '-I'+str(ROOT/'third_party/xxHash'),str(tmp/'hash.cpp'),
                        str(tmp/'bridge.cpp'),str(ROOT/'third_party/xxHash/xxhash.c'),
                        '-o',str(lib)],check=True,capture_output=True)
        native=ctypes.CDLL(str(lib));native.hnew.restype=ctypes.c_void_p
        native.hfree.argtypes=[ctypes.c_void_p];native.hreset.argtypes=[ctypes.c_void_p]
        native.hdigest.argtypes=[ctypes.c_void_p];native.hdigest.restype=ctypes.c_uint64
        native.hupdate.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
        native.hgen.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_void_p]
        class Hasher:
            def __init__(self):self.ptr=native.hnew()
            def __del__(self):native.hfree(self.ptr)
            def reset(self):native.hreset(self.ptr)
            def update(self,a):native.hupdate(self.ptr,a.ctypes.data,a.nbytes)
            def digest(self):return native.hdigest(self.ptr)
        def gen(h,a,page,out):native.hgen(h.ptr,a.ctypes.data,a.size,page,out.ctypes.data)
        ext=NS(Hasher=Hasher,gen_hashes=gen,get_hash_size=lambda:8)
        hashmod=load_nodes(ROOT/'flexkv/common/hash_utils.py',{'Hasher','gen_hashes','get_hash_size'},
            dict(base,c_ext=ext,torch=NS(from_numpy=lambda x:x),HashType=int))
        blockmod=load_nodes(ROOT/'flexkv/common/block.py',{'SequenceMeta','_get_namespace_hash_key'},
                           dict(base,HashType=int,Hasher=hashmod.Hasher,gen_hashes=hashmod.gen_hashes,get_hash_size=lambda:8))
        cls.SequenceMeta=blockmod.SequenceMeta

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def chain(self,tokens,page,extra=None):
        ids=codec.encode_bigram_ids(tokens,len(tokens)-1)
        return self.SequenceMeta(ids,page,codec.cache_namespace(extra,True)).block_hashes

    def test_page_boundary_token_changes_this_page_and_all_following_pages(self):
        for page in (1,4,64,128):
            raw=np.arange(page*3+1,dtype=np.int64)
            first=self.chain(raw,page)
            changed=raw.copy();changed[page]+=123456
            second=self.chain(changed,page)
            self.assertTrue(np.all(first!=second))
            tail=raw.copy();tail[page*3]+=1
            last=self.chain(tail,page)
            np.testing.assert_array_equal(first[:2],last[:2])
            self.assertNotEqual(first[2],last[2])

    def test_prefix_store_lookup_prefetch_share_identical_native_hash_chain(self):
        raw=np.arange(193,dtype=np.int64)
        full=self.chain(raw,64,'image:a\0b')
        for pages in (1,2,3):
            np.testing.assert_array_equal(full[:pages],self.chain(raw[:pages*64+1],64,'image:a\0b'))
        self.assertTrue(np.all(full!=self.chain(raw,64,'image:b')))

    def test_signed_int64_pair_bits_survive_real_native_hasher(self):
        raw=np.array([0xffffffff,0xffffffff,0x80000000,1,2],dtype=np.uint64)
        packed=codec.encode_bigram_ids(raw,4)
        self.assertLess(packed[0],0)
        expected=self.chain(raw,1)
        np.testing.assert_array_equal(expected,self.chain(raw.copy(),1))
        changed=raw.copy();changed[1]-=1
        self.assertTrue(np.all(expected!=self.chain(changed,1)))

    def test_actual_mooncake_adapter_round_trips_full_group_bytes_with_inline_scales(self):
        # One CP8 block: target KV + sparse Indexer + each rank's draft+Indexer.
        sizes=[78*64*656,21*8448]+[64*656,8448]*8
        rng=np.random.default_rng(21)
        source=rng.integers(0,256,size=sum(sizes),dtype=np.uint8)
        target=np.zeros_like(source);objects={}
        client=MooncakeStoreClient.__new__(MooncakeStoreClient)
        import threading
        client._get_lock=threading.Lock();client._get_closed=False
        client._get_concurrency=1;client._get_executor=None
        client._is_setup=True
        client._config=NS(max_get_batch_bytes=0,max_put_batch_bytes=0)
        client._get_max_inflight_bytes=512*1024*1024
        def put(keys,ptrs,counts):
            for k,p,n in zip(keys,ptrs,counts):objects[k]=ctypes.string_at(p,n)
            return [0]*len(keys)
        def get(keys,ptrs,counts):
            for k,p,n in zip(keys,ptrs,counts):ctypes.memmove(p,objects[k],n)
            return counts
        client._store=NS(batch_put_from=put,batch_get_into=get,
                         batch_is_exist=lambda keys:[int(k in objects) for k in keys])
        key=str(int(self.chain(np.arange(65),64)[0]))
        self.assertEqual(client.batch_put([key],[source.ctypes.data],[source.nbytes]),[True])
        self.assertEqual(client.batch_get([key],[target.ctypes.data],[target.nbytes]),[True])
        np.testing.assert_array_equal(source,target)
        # Distinct rank rows remain distinct; opaque scale bytes are included.
        offset=sizes[0]+sizes[1]
        for owner in range(8):
            end=offset+64*656+8448
            np.testing.assert_array_equal(source[offset:end],target[offset:end]);offset=end

if __name__=='__main__':unittest.main()
