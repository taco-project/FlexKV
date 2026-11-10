"""Real Mooncake client byte IO under concurrent callers and injected errors."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import threading
import pytest
import numpy as np
from mooncake_test_support import MooncakeStoreClient,MooncakeStoreConfig
from mooncake_byte_store_test_support import ByteStore,guarded_buffers,assert_guards


def client(store,readers,sizes):
    obj=MooncakeStoreClient.__new__(MooncakeStoreClient)
    obj._store=store;obj._is_setup=True;obj._get_closed=False
    obj._get_lock=threading.Lock();obj._get_concurrency=readers
    obj._get_max_inflight_bytes=readers*max(((n+4095)//4096)*4096+8192 for n in sizes)*2
    obj._get_executor=ThreadPoolExecutor(max_workers=readers) if readers>1 else None
    obj._config=MooncakeStoreConfig(max_get_batch_bytes=max(sizes)*2,
                                    max_put_batch_bytes=max(sizes)*2)
    return obj


@pytest.mark.parametrize('readers',[1,2,8])
@pytest.mark.parametrize('size',[1,4095,4096,4097,65535,65536,65537,1048577])
def test_mixed_object_sizes_guarded_roundtrip_and_order(readers,size):
    sizes=[size,1,max(1,size-1),size+1,17,4096,1]
    store=ByteStore();arrays,ptrs,values=guarded_buffers(store,sizes)
    c=client(store,readers,sizes);keys=[f'object-{i}' for i in range(len(sizes))]
    try:
        for pointer,value in zip(ptrs,values):ctypes.memmove(pointer,value,len(value))
        assert c.batch_put(keys,ptrs,sizes)==[True]*len(sizes)
        for array in arrays:array[33:-33]=0
        store.delay=.0001
        assert c.batch_get(keys,ptrs,sizes)==[True]*len(sizes)
        for pointer,size,value in zip(ptrs,sizes,values):assert ctypes.string_at(pointer,size)==value
        assert_guards(arrays)
        assert store.active==0 and store.peak<=readers
        # Aggregate padded credit belongs to parallel Get; serial default
        # intentionally retains the raw-byte SDK batching contract.
        if readers>1:assert store.peak_bytes<=c._get_max_inflight_bytes
        assert all(sum(n)<=max(sizes)*2 for op,_,n in store.calls)
    finally:c.shutdown_get_workers()


@pytest.mark.parametrize('readers',[1,2,8])
@pytest.mark.parametrize('callers',[2,8,32])
def test_many_shared_client_callers_keep_independent_bytes_and_global_credit(readers,callers):
    store=ByteStore();sizes=[1,4095,4096,4097,8193,17,65537]
    c=client(store,readers,sizes);store.delay=.0001
    def actor(index):
        arrays,ptrs,values=guarded_buffers(store,sizes,seed=index+10)
        keys=[f'req-{index}-block-{i}' for i in range(len(sizes))]
        for ptr,value in zip(ptrs,values):ctypes.memmove(ptr,value,len(value))
        assert c.batch_put(keys,ptrs,sizes)==[True]*len(keys)
        for _ in range(3):
            assert c.batch_get(keys,ptrs,sizes)==[True]*len(keys)
            assert all(ctypes.string_at(ptr,n)==v for ptr,n,v in zip(ptrs,sizes,values))
            assert_guards(arrays)
    try:
        with ThreadPoolExecutor(max_workers=callers) as pool:
            futures=[pool.submit(actor,i) for i in range(64)]
            for f in futures:f.result(timeout=15)
        assert store.active==0 and store.peak<=readers
        if readers>1:assert store.peak_bytes<=c._get_max_inflight_bytes
        assert len(store.objects)==64*len(sizes)
    finally:c.shutdown_get_workers()


@pytest.mark.parametrize('readers',[1,2,8])
@pytest.mark.parametrize('failure',['missing','zero','short','oversize','exception'])
def test_read_failure_drain_guards_and_same_buffer_retry(readers,failure):
    store=ByteStore();sizes=[4097]*17
    arrays,ptrs,values=guarded_buffers(store,sizes)
    keys=[f'block-{i}' for i in range(17)]
    store.objects.update(zip(keys,values));store.delay=.0001
    c=client(store,readers,sizes)
    def fault(key,index,size):
        if key==keys[3]:
            if failure=='exception':raise RuntimeError('SDK fault after peer writes')
            return {'missing':-1,'zero':0,'short':size-1,'oversize':size+1}[failure]
        return size
    store.get_hook=fault
    try:
        if failure=='exception':
            with pytest.raises(RuntimeError,match='SDK fault'):c.batch_get(keys,ptrs,sizes)
        else:
            result=c.batch_get(keys,ptrs,sizes)
            assert result==[i!=3 for i in range(17)]
        assert store.active==0
        assert_guards(arrays)
        # Retry must overwrite every byte, regardless of previous partial work.
        store.get_hook=None
        assert c.batch_get(keys,ptrs,sizes)==[True]*17
        assert all(ctypes.string_at(p,n)==v for p,n,v in zip(ptrs,sizes,values))
        assert_guards(arrays)
    finally:c.shutdown_get_workers()


@pytest.mark.parametrize('failure',['status','exception','short_results','none_results'])
def test_partial_put_retry_preserves_existing_objects(failure):
    store=ByteStore();sizes=[4097]*9
    arrays,ptrs,values=guarded_buffers(store,sizes)
    for ptr,value in zip(ptrs,values):ctypes.memmove(ptr,value,len(value))
    keys=[f'block-{i}' for i in range(9)];c=client(store,1,sizes)
    store.objects[keys[0]]=values[0]
    original=store.batch_put_from
    def faulty(keys_,ptrs_,sizes_):
        result=original(keys_,ptrs_,sizes_)
        if failure=='exception':raise RuntimeError('partial accepted put')
        if failure=='short_results':return result[:-1]
        if failure=='none_results':return None
        return result
    if failure=='status':store.put_hook=lambda key,i:-1 if key==keys[4] else 0
    else:store.batch_put_from=faulty
    try:
        if failure=='status':assert c.batch_put(keys,ptrs,sizes)==[i!=4 for i in range(9)]
        else:
            with pytest.raises(RuntimeError):c.batch_put(keys,ptrs,sizes)
        store.batch_put_from=original;store.put_hook=None
        assert c.batch_put(keys,ptrs,sizes)==[True]*9
        assert [store.objects[k] for k in keys]==values
        assert all(keys[0] not in batch for op,batch,_ in store.calls if op=='put')
        assert_guards(arrays)
    finally:c.shutdown_get_workers()


@pytest.mark.parametrize('readers',[1,2,8])
def test_close_after_many_reads_rejects_all_new_writes_to_destination(readers):
    store=ByteStore();sizes=[4097]*9
    arrays,ptrs,values=guarded_buffers(store,sizes);keys=list(map(str,range(9)))
    store.objects.update(zip(keys,values));c=client(store,readers,sizes)
    assert c.batch_get(keys,ptrs,sizes)==[True]*9
    c.shutdown_get_workers();c.shutdown_get_workers()
    snapshot=[a.copy() for a in arrays];calls=len(store.calls)
    with pytest.raises(RuntimeError,match='closed'):c.batch_get(keys,ptrs,sizes)
    assert len(store.calls)==calls and store.active==0
    for a,b in zip(arrays,snapshot):np.testing.assert_array_equal(a,b)


if __name__=='__main__':raise SystemExit(pytest.main(['--noconftest','-q',__file__]))
