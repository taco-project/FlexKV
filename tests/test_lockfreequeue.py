#!/usr/bin/env python3
import os
import sys
import time
import threading

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flexkv import c_ext


def test_push_pop_basic():
    q = c_ext.IntQueue()
    # empty pop
    ok, val = q.pop()
    assert ok is False

    # push then pop
    q.push(42)
    ok, val = q.pop()
    assert ok is True and val == 42


def test_fifo_order():
    q = c_ext.IntQueue()
    for i in range(10):
        q.push(i)
    out = []
    while True:
        ok, v = q.pop()
        if not ok:
            break
        out.append(v)
    assert out == list(range(10))


def test_multithread_producer_consumer():
    q = c_ext.IntQueue()
    N = 1000
    NUM_PROD = 4
    NUM_CONS = 4

    produced = []
    consumed = []
    plock = threading.Lock()
    clock = threading.Lock()

    def producer(start):
        print(f"[Producer {start//N}] Starting, range {start}-{start+N-1}")
        for i in range(start, start + N):
            q.push(i)
            with plock:
                produced.append(i)
            if i % 100 == 0:  # Print every 100 items
                print(f"[Producer {start//N}] Pushed {i}")
        print(f"[Producer {start//N}] Finished")

    def consumer():
        local = []
        end_time = time.time() + 2.0
        consumer_id = threading.current_thread().ident
        print(f"[Consumer {consumer_id}] Starting")
        while time.time() < end_time:
            ok, v = q.pop()
            if ok:
                local.append(v)
                if len(local) % 100 == 0:  # Print every 100 items
                    print(f"[Consumer {consumer_id}] Popped {v}, total consumed: {len(local)}")
            else:
                time.sleep(0.0005)
        with clock:
            consumed.extend(local)
        print(f"[Consumer {consumer_id}] Finished, consumed {len(local)} items")

    threads = []
    for p in range(NUM_PROD):
        t = threading.Thread(target=producer, args=(p * N,))
        t.start()
        threads.append(t)
    for c in range(NUM_CONS):
        t = threading.Thread(target=consumer)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    print(f"Total produced: {len(produced)}, Total consumed: {len(consumed)}")
    # Since consumers stop after timeout, they may not drain completely.
    # Verify no duplicates and subset correctness.
    assert len(set(consumed)) == len(consumed)
    assert set(consumed).issubset(set(produced))


if __name__ == "__main__":
    # Simple manual run
    test_push_pop_basic()
    test_fifo_order()
    test_multithread_producer_consumer()
    print("All IntQueue tests passed")


