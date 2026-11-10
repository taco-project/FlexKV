"""Load real Mooncake adapter classes with no SDK, torch or native import."""
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
import threading
from submission_test_support import ROOT, NS, base, load_nodes, noop

namespace = dict(base, os=os, time=time, json=json, uuid=uuid,
    threading=threading, ThreadPoolExecutor=ThreadPoolExecutor,
    put_trace=NS(ENABLED=False, mark=noop, volume=noop),
    DEFAULT_LOCAL_BUFFER_SIZE=16*1024*1024, SETUP_TIMEOUT=600,
    torch=NS(Tensor=type('Tensor', (), {})))
module = load_nodes(ROOT/'flexkv/external/mooncake_store_utils.py',
    {'MooncakeStoreConfig', 'MooncakeStoreClient'}, namespace)
MooncakeStoreConfig = module.MooncakeStoreConfig
MooncakeStoreClient = module.MooncakeStoreClient
