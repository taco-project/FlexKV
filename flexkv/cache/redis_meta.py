from typing import Iterable, List, Tuple, Optional, Union, Dict
from dataclasses import dataclass
from enum import IntEnum
from uuid import uuid1
import threading
import time
import atexit
import signal
import sys
try:  # redis-py
    import redis as _redis
except Exception:  # pragma: no cover
    _redis = None  # type: ignore

# Import C++ dist extensions (RedisMetaChannel, BlockMeta). Optional when built without FLEXKV_ENABLE_P2P=1.
_CRedisMetaChannel = None  # type: ignore
_CBlockMeta = None  # type: ignore
try:
    import flexkv.c_ext
    from flexkv.c_ext import RedisMetaChannel as _CRedisMetaChannel, BlockMeta as _CBlockMeta  # type: ignore
except (ImportError, AttributeError):
    # c_ext built without FLEXKV_ENABLE_P2P=1: no Redis/distributed KV cache support
    pass


class NodeState(IntEnum):
    NODE_STATE_NORMAL = 0
    NODE_STATE_ABOUT_TO_EVICT = 1
    NODE_STATE_EVICTED = 2


@dataclass
class BlockMeta:
    ph: int = 0
    pb: int = 0
    nid: int = 0
    hash: int = 0
    lt: int = 0
    state: NodeState = NodeState.NODE_STATE_NORMAL

    def to_c(self) -> "_CBlockMeta":
        if _CBlockMeta is None:
            raise RuntimeError(
                "Distributed KV cache (P2P/Redis) is not built. "
                "Rebuild FlexKV with FLEXKV_ENABLE_P2P=1 and install Redis dependencies (e.g. libhiredis-dev)."
            )
        cm = _CBlockMeta()
        cm.ph = int(self.ph)
        cm.pb = int(self.pb)
        cm.nid = int(self.nid)
        cm.hash = int(self.hash)
        cm.lt = int(self.lt)
        cm.state = int(self.state)
        return cm

    @staticmethod
    def from_c(cm: "_CBlockMeta") -> "BlockMeta":
        if _CBlockMeta is None:
            raise RuntimeError(
                "Distributed KV cache (P2P/Redis) is not built. "
                "Rebuild FlexKV with FLEXKV_ENABLE_P2P=1 and install Redis dependencies (e.g. libhiredis-dev)."
            )
        return BlockMeta(
            ph=int(cm.ph),
            pb=int(cm.pb),
            nid=int(cm.nid),
            hash=int(cm.hash),
            lt=int(cm.lt),
            state=NodeState(int(cm.state))
        )


def dist_available() -> bool:
    """Return True if distributed (P2P/Redis) KV cache C++ extension is built (FLEXKV_ENABLE_P2P=1)."""
    return _CRedisMetaChannel is not None


class RedisMetaChannel:
    def __init__(self, host: str, port: int, node_id: int, local_ip: str, blocks_key: str = "blocks", password: str = "") -> None:
        if _CRedisMetaChannel is None:
            raise ImportError(
                "Distributed KV cache (P2P/Redis) is not built. "
                "Rebuild FlexKV with FLEXKV_ENABLE_P2P=1 and install Redis dependencies (e.g. libhiredis-dev, redis-tools)."
            )
        self._c = _CRedisMetaChannel(host, int(port), int(node_id), str(local_ip), str(blocks_key), str(password))

    def connect(self) -> bool:
        return bool(self._c.connect())

    @property
    def node_id(self) -> int:
        return int(self._c.get_node_id())

    @property
    def local_ip(self) -> str:
        return str(self._c.get_local_ip())

    def make_block_key(self, node_id: int, hash_value: int) -> str:
        return str(self._c.make_block_key(int(node_id), int(hash_value)))

    def publish_one(self, meta: BlockMeta) -> bool:
        """publish single BlockMeta to Redis"""
        return self._c.publish_one(meta.to_c())

    def publish_batch(self, metas: Iterable[BlockMeta], batch_size: int = 100) -> bool:
        """batch publish BlockMeta to Redis"""
        cms = [m.to_c() for m in metas]
        return self._c.publish_batch(cms, int(batch_size))

    def list_keys(self, pattern: str) -> List[str]:
        return list(self._c.list_keys(pattern))

    def list_node_keys(self) -> List[str]:
        return list(self._c.list_node_keys())

    def list_block_keys(self, node_id: int) -> List[str]:
        return list(self._c.list_block_keys(int(node_id)))

    def hmget_field_for_keys(self, keys: Iterable[str], field: str) -> List[str]:
        return list(self._c.hmget_field_for_keys(list(keys), field))

    def hmget_two_fields_for_keys(self, keys: Iterable[str], f1: str, f2: str) -> List[Tuple[str, str]]:
        return [(a, b) for a, b in self._c.hmget_two_fields_for_keys(list(keys), f1, f2)]

    def renew_node_leases(self, node_id: int, new_lt: int, batch_size: int = 200) -> bool:
        """batch update lease time for specified node"""
        return self._c.renew_node_leases(int(node_id), int(new_lt), int(batch_size))

    def update_block_state_batch(self, node_id: int, hashes: Iterable[int], state: int, batch_size: int = 200) -> bool:
        """batch update block state for specified node"""
        return self._c.update_block_state_batch(int(node_id), list(int(h) for h in hashes), int(state), int(batch_size))

    def delete_blockmeta_batch(self, node_id: int, hashes: Iterable[int], batch_size: int = 200) -> bool:
        """batch delete block metadata for specified node"""
        return self._c.delete_blockmeta_batch(int(node_id), list(int(h) for h in hashes), int(batch_size))

class RedisNodeInfo:
    """Redis node information management class implemented in Python"""

    # Default TTL for node:<id> key in seconds. Active nodes renew before expiry.
    # If a process crashes (kill -9), the key auto-expires after this period.
    DEFAULT_NODE_TTL_SECONDS: int = 30
    
    def __init__(self, host: str, port: int, local_ip: str, password: str = "", node_ttl_seconds: int = 0) -> None:
        if _redis is None:
            raise ImportError("redis-py is required: pip install redis")
        self.host = host
        self.port = int(port)
        self.local_ip = str(local_ip)
        self.password = str(password)
        self.uuid = str(uuid1())
        # Use provided TTL or fall back to default
        self.node_ttl_seconds: int = node_ttl_seconds if node_ttl_seconds > 0 else self.DEFAULT_NODE_TTL_SECONDS
        # Heartbeat interval – renew TTL at roughly 1/3 of the TTL period
        self.heartbeat_interval_seconds: float = max(1.0, self.node_ttl_seconds / 3.0)
        self._node_id: Optional[int] = None
        self._running = False
        self._listener_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self.current_node_id_set: set = set()
        self._client: Optional[_redis.Redis] = None
        self._sub_client: Optional[_redis.Redis] = None
        self._cleanup_done = False
        
        # register cleanup function on exit
        atexit.register(self._cleanup_on_exit)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def __del__(self) -> None:
        """destructor, ensure cleanup is performed when object is destroyed"""
        try:
            self._cleanup_on_exit()
        except Exception:
            # ignore exceptions in destructor, avoid affecting program exit
            pass
    
    def _get_client(self) -> _redis.Redis:
        """Get Redis client with connection settings"""
        return _redis.Redis(
            host=self.host,
            port=self.port,
            password=self.password if self.password else None,
            decode_responses=True,
            health_check_interval=30,
            socket_keepalive=True
        )
    
    def connect(self) -> bool:
        """Connect to Redis and start listener + heartbeat threads"""
        try:
            self._client = self._get_client()
            # Test connection
            self._client.ping()
            
            # Start listener thread
            self._running = True
            self._listener_thread = threading.Thread(
                target=self._listener_worker,
                name="redis-node-info-listener",
                daemon=True
            )
            self._listener_thread.start()

            # Start heartbeat thread for TTL renewal
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_worker,
                name="redis-node-heartbeat",
                daemon=True
            )
            self._heartbeat_thread.start()
            
            return True
        except Exception:
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Redis and stop listener + heartbeat threads"""
        self._running = False
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2.0)
        self._listener_thread = None

        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2.0)
        self._heartbeat_thread = None
        
        if self._client:
            self._client.close()
            self._client = None
        if self._sub_client:
            self._sub_client.close()
            self._sub_client = None
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Signal handler for graceful shutdown"""
        print(f"received signal {signum}, starting cleanup of RedisNodeInfo...")
        self._cleanup()
        sys.exit(0)
    
    def _cleanup_on_exit(self) -> None:
        """Cleanup function registered with atexit"""
        self._cleanup()
    
    def _cleanup(self) -> None:
        """Internal cleanup method"""
        if self._cleanup_done:
            return
        
        self._cleanup_done = True
        
        try:
            # unregister node
            if self._node_id is not None:
                self.unregister_node()
            
            # disconnect
            self.disconnect()
        except Exception:
            # ignore exceptions in cleanup
            pass
    
    def register_node(self) -> Optional[int]:
        """Register a new node and get node_id, with TTL for automatic expiry on crash"""
        if not self._client:
            return None
        
        try:
            # Clean up stale nodes from the same IP before registering
            self._cleanup_stale_nodes_by_ip()

            # Atomically increment global:node_id to get new node_id
            node_id = self._client.incr("global:node_id")
            self._node_id = node_id
            
            # Store node information in node:node_id hash
            node_key = f"node:{node_id}"
            self._client.hset(node_key, mapping={
                "node_id": str(node_id),
                "ip": self.local_ip,  # Changed from "local_ip" to "ip" to match C++ code expectation
                "local_ip": self.local_ip,  # Keep for backward compatibility
                "uuid": self.uuid,
                "status": "active",
                "timestamp": str(int(time.time()))
            })

            # Set TTL so the key auto-expires if the process crashes
            self._client.expire(node_key, self.node_ttl_seconds)
            
            # Publish node update event
            self._client.publish("flexkv_node_id_updated", str(node_id))
            
            return node_id
        except Exception:
            return None
    
    def unregister_node(self) -> bool:
        """Unregister current node and clean up associated meta/block data"""
        if not self._client or self._node_id is None:
            return False
        
        try:
            node_id = self._node_id

            # Delete node:node_id key
            node_key = f"node:{node_id}"
            self._client.delete(node_key)

            # Also clean up meta:<node_id> to prevent stale RDMA addresses
            self._cleanup_node_data(node_id)
            
            # Publish node update event
            self._client.publish("flexkv_node_id_updated", str(node_id))
            
            self._node_id = None
            return True
        except Exception:
            return False
    
    @property
    def node_id(self) -> Optional[int]:
        """Get current node_id"""
        return self._node_id
    
    def get_uuid(self) -> str:
        """Get the UUID of this node"""
        return self.uuid
    
    def get_active_node_ids(self) -> List[int]:
        """Get all active node IDs - lock-free RCU read"""
        return list(self.current_node_id_set)
    
    def is_node_active(self, node_id: int) -> bool:
        """Check if a node_id is active - lock-free RCU check"""
        return node_id in self.current_node_id_set
    
    def _heartbeat_worker(self) -> None:
        """Background thread that periodically renews the TTL of node:<id> key.
        
        This ensures that if the process is alive, the node key never expires.
        If the process crashes (kill -9), the TTL will not be renewed and the
        key will auto-expire after NODE_TTL_SECONDS, allowing other nodes to
        detect the crash and stop using stale meta/block data.
        """
        heartbeat_client: Optional["_redis.Redis"] = None
        while self._running:
            try:
                if heartbeat_client is None:
                    heartbeat_client = self._get_client()

                if self._node_id is not None:
                    node_key = f"node:{self._node_id}"
                    # Renew TTL
                    heartbeat_client.expire(node_key, self.node_ttl_seconds)
                    # Also update the timestamp field
                    heartbeat_client.hset(node_key, "timestamp", str(int(time.time())))

            except Exception:
                # Connection lost, reset client so it reconnects next iteration
                if heartbeat_client:
                    try:
                        heartbeat_client.close()
                    except Exception:
                        pass
                    heartbeat_client = None

            # Sleep in small increments so we can exit quickly when _running becomes False
            for _ in range(int(self.heartbeat_interval_seconds * 10)):
                if not self._running:
                    break
                time.sleep(0.1)

        if heartbeat_client:
            try:
                heartbeat_client.close()
            except Exception:
                pass

    def _listener_worker(self) -> None:
        """Background thread that listens for node updates"""
        backoff = 0.5
        while self._running:
            try:
                # Create a separate connection for pub/sub
                self._sub_client = self._get_client()
                
                # Subscribe to flexkv_node_id_updated channel
                pubsub = self._sub_client.pubsub()
                pubsub.subscribe("flexkv_node_id_updated")
                
                # Listen for messages with blocking read
                for message in pubsub.listen():
                    if not self._running:
                        break
                    
                    if message["type"] == "message" and message["channel"] == "flexkv_node_id_updated":
                        # Scan active nodes when we receive an update
                        self.scan_active_nodes()
                
                # Normal exit from listen loop
                break
                
            except Exception:
                # Network/reconnection exception: exponential backoff
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
            finally:
                if self._sub_client:
                    try:
                        self._sub_client.close()
                    except Exception:
                        pass
                self._sub_client = None
    
    def scan_active_nodes(self) -> None:
        """Scan Redis for active node keys and update current_node_id_set
        
        This method can be called externally to manually refresh the active nodes list.
        It uses SCAN to avoid blocking Redis server.
        
        Because node:<id> keys now have a TTL (heartbeat), expired keys are
        automatically removed by Redis.  SCAN will only return keys that are
        still alive, so stale/crashed nodes are naturally excluded.
        """
        if not self._client:
            return
        
        try:
            new_active_nodes = set()
            cursor = 0
            
            while True:
                cursor, keys = self._client.scan(cursor=cursor, match="node:*", count=100)
                
                for key in keys:
                    if key.startswith("node:"):
                        try:
                            node_id = int(key[5:])  # Remove "node:" prefix
                            new_active_nodes.add(node_id)
                        except (ValueError, IndexError):
                            # Skip invalid node IDs
                            continue
                
                if cursor == 0:
                    break
            
            # Detect nodes that disappeared (TTL expired or unregistered)
            disappeared = self.current_node_id_set - new_active_nodes
            if disappeared:
                # Clean up meta and block data for disappeared nodes
                for stale_nid in disappeared:
                    if stale_nid == self._node_id:
                        continue  # Don't clean up ourselves
                    self._cleanup_node_data(stale_nid)

            # lock-free RCU switch: atomic assignment
            self.current_node_id_set = new_active_nodes
                
        except Exception:
            # If scan fails, continue with current active nodes
            pass

    def _cleanup_stale_nodes_by_ip(self) -> None:
        """Clean up stale node registrations from the same IP.
        
        On startup, scan all node:* keys and remove those that have the same
        local_ip but a different UUID (i.e. leftover from a previous crashed process).
        """
        if not self._client:
            return

        try:
            cursor = 0
            stale_node_ids = []

            while True:
                cursor, keys = self._client.scan(cursor=cursor, match="node:*", count=100)
                for key in keys:
                    if not key.startswith("node:"):
                        continue
                    try:
                        nid = int(key[5:])
                    except (ValueError, IndexError):
                        continue

                    data = self._client.hgetall(key)
                    node_ip = data.get("ip", "") or data.get("local_ip", "")
                    node_uuid = data.get("uuid", "")

                    # Same IP but different UUID → stale node from a previous process
                    if node_ip == self.local_ip and node_uuid != self.uuid:
                        stale_node_ids.append(nid)

                if cursor == 0:
                    break

            for stale_nid in stale_node_ids:
                print(f"[RedisNodeInfo] Cleaning up stale node:{stale_nid} (same IP={self.local_ip}, different UUID)")
                self._client.delete(f"node:{stale_nid}")
                self._cleanup_node_data(stale_nid)

            if stale_node_ids:
                # Notify other nodes about the cleanup
                self._client.publish("flexkv_node_id_updated", "cleanup")

        except Exception:
            pass

    def _cleanup_node_data(self, node_id: int) -> None:
        """Clean up meta:<node_id> and CPUB/SSDB/PCFSB block keys for a given node.
        
        This is called when:
        1. A node is unregistered (graceful shutdown)
        2. A stale node is detected (TTL expired / startup cleanup)
        """
        if not self._client:
            return

        try:
            # Delete meta:<node_id> (and meta:<node_id>:pp* for pipeline parallel)
            cursor = 0
            meta_keys = []
            while True:
                cursor, keys = self._client.scan(cursor=cursor, match=f"meta:{node_id}*", count=100)
                meta_keys.extend(keys)
                if cursor == 0:
                    break
            if meta_keys:
                self._client.delete(*meta_keys)
                print(f"[RedisNodeInfo] Deleted {len(meta_keys)} meta key(s) for node {node_id}")

            # Delete CPUB:block:<node_id>:* / SSDB:block:<node_id>:* / PCFSB:block:<node_id>:* keys
            for prefix in ("CPUB", "SSDB", "PCFSB"):
                cursor = 0
                block_keys = []
                while True:
                    cursor, keys = self._client.scan(cursor=cursor, match=f"{prefix}:block:{node_id}:*", count=500)
                    block_keys.extend(keys)
                    if cursor == 0:
                        break
                if block_keys:
                    # Delete in batches to avoid blocking Redis
                    batch_size = 500
                    for i in range(0, len(block_keys), batch_size):
                        self._client.delete(*block_keys[i:i + batch_size])
                    print(f"[RedisNodeInfo] Deleted {len(block_keys)} {prefix}:block key(s) for node {node_id}")

        except Exception as e:
            print(f"[RedisNodeInfo] Warning: failed to clean up data for node {node_id}: {e}")


class RedisMeta:
    def __init__(self, host: str, port: int, password: Optional[str] = None, local_ip: str = "127.0.0.1", decode_responses: bool = True, node_ttl_seconds: int = 0) -> None:
        if _redis is None:  # pragma: no cover
            raise ImportError("redis-py is required: pip install redis")
        self.host = host
        self.port = int(port)
        self.local_ip = str(local_ip)
        self.db = 0
        self.password = password
        self.decode_responses = bool(decode_responses)
        self._node_id: Optional[int] = None
        
        # initialize state management
        self._init_lock = threading.Lock()
        self._initialized = False
        self._init_error: Optional[Exception] = None
        
        # create RedisNodeInfo object
        self.nodeinfo = RedisNodeInfo(host, port, local_ip, password or "", node_ttl_seconds=node_ttl_seconds)
        # get UUID via nodeinfo
        self._uuid = self.nodeinfo.get_uuid()

    def _client(self):
        return _redis.Redis(host=self.host, port=self.port, db=self.db, password=self.password, decode_responses=self.decode_responses)

    def init_meta(self) -> Optional[int]:
        """Initialize Redis metadata. This method is thread-safe and can only be called once per instance.
        
        Returns:
            Optional[int]: The registered node ID, or None if initialization fails
            
        Raises:
            RuntimeError: If initialization fails or has already been called
        """
        with self._init_lock:
            # check if already initialized
            if self._initialized:
                if self._init_error:
                    raise self._init_error
                return self._node_id
            
            try:
                # connect to RedisNodeInfo
                if not self.nodeinfo.connect():
                    raise RuntimeError("Failed to connect to Redis via RedisNodeInfo")
                
                # register node
                node_id = self.nodeinfo.register_node()
                if node_id is None:
                    raise RuntimeError("Failed to register node via RedisNodeInfo")
                
                self._node_id = node_id
                # initialization phase, scan active nodes first
                self.nodeinfo.scan_active_nodes()
                
                # mark as initialized
                self._initialized = True
                
                return node_id
                
            except Exception as e:
                # record initialization error
                self._init_error = e
                return None

    def get_node_id(self) -> int:
        if self._node_id is None:
            raise RuntimeError("node_id is not registered yet. Call init_meta() first.")
        return int(self._node_id)
    
    def is_initialized(self) -> bool:
        """Check if RedisMeta has been initialized.
        
        Returns:
            bool: True if initialized, False otherwise
        """
        with self._init_lock:
            return self._initialized
    
    def get_init_error(self) -> Optional[Exception]:
        """Get the initialization error if any.
        
        Returns:
            Optional[Exception]: The initialization error, or None if no error
        """
        with self._init_lock:
            return self._init_error

    def get_redis_meta_channel(self, blocks_key: str = "blocks") -> "RedisMetaChannel":
        nid = self.get_node_id()
        # Avoid passing string "None" when no password is set
        pwd = "" if (self.password is None or str(self.password).lower() == "none") else str(self.password)
        channel = RedisMetaChannel(self.host, int(self.port), int(nid), self.local_ip, str(blocks_key), pwd)
        if not channel.connect():
            raise RuntimeError("Failed to connect to Redis")
        return channel

    def unregister_node(self, node_id: Optional[int] = None) -> None:
        # use RedisNodeInfo to unregister node
        if self.nodeinfo:
            self.nodeinfo.unregister_node()
        self._node_id = None

    def get_uuid(self) -> str:
        return self._uuid
    
    def get_active_node_ids(self) -> List[int]:
        """get all active node IDs list"""
        if self.nodeinfo:
            return self.nodeinfo.get_active_node_ids()
        return []
    
    def is_node_active(self, node_id: int) -> bool:
        """check if specified node is active"""
        if self.nodeinfo:
            return self.nodeinfo.is_node_active(node_id)
        return False

    def add_node_ids(self, node_ids: Iterable[Union[int, str]]) -> int:
        # Append a list of pcfs file node ids to Redis list key pcfs:<node_id>
        nid = self.get_node_id()
        values = [str(v) for v in node_ids]
        if not values:
            return 0
        r = self._client()
        # rpush returns the new length of the list
        return int(r.rpush(f"pcfs:{nid}", *values))

    def regist_buffer(self, mrs: Iterable[object]) -> int:
        """Register RDMA memory regions in Redis.

        Each element in mrs can be one of:
          - dict with keys {"buffer_ptr": ..., "buffer_size": ...}
          - tuple/list (buffer_ptr, buffer_size)
        Stored as hash: key = buffer:<node_id>:<buffer_ptr>, field "buffer_size" = <buffer_size>.
        Returns the number of regions processed.
        """
        nid = self.get_node_id()
        r = self._client()
        pipe = r.pipeline()
        processed = 0
        for mr in mrs:
            if isinstance(mr, dict):
                ptr = mr.get("buffer_ptr")
                size = mr.get("buffer_size")
            elif isinstance(mr, (tuple, list)) and len(mr) >= 2:
                ptr, size = mr[0], mr[1]
            else:
                continue
            if ptr is None or size is None:
                continue
            key = f"buffer:{nid}:{int(ptr)}"
            pipe.hset(key, mapping={"buffer_size": int(size)})
            processed += 1
        if processed:
            pipe.execute()
        return processed

    def unregist_buffer(self, buffer_ptr: Union[int, str]) -> bool:
        """Unregister a previously registered RDMA memory region by buffer_ptr.

        Looks up key buffer:<node_id>:<buffer_ptr> and deletes it if present.
        Returns True if the key existed and was deleted, otherwise False.
        """
        nid = self.get_node_id()
        key = f"buffer:{nid}:{int(buffer_ptr)}"
        r = self._client()
        exists = bool(r.exists(key))
        if exists:
            r.delete(key)
            return True
        return False

    def regist_node_meta(self, node_id: int, addr: str, zmq_addr: str, cpu_buffer_ptr: int, ssd_buffer_ptr: int) -> None:
        """Register node meta information as a Redis hash.

        Key: meta:<node_id>
        Fields: node_id (int), addr (str), cpu_buffer_ptr (int), ssd_buffer_ptr (int)
        """
        r = self._client()
        key = f"meta:{int(node_id)}"
        r.hset(key, mapping={
            "node_id": int(node_id),
            "addr": str(addr),
            "zmq_addr": str(zmq_addr),
            "cpu_buffer_ptr": int(cpu_buffer_ptr),
            "ssd_buffer_ptr": int(ssd_buffer_ptr),
        })

    def get_node_meta(self, node_id: int) -> dict:
        """Get node meta information from Redis.

        Reads key meta:<node_id> and returns a dict with fields:
        node_id (int), addr (str), cpu_buffer_ptr (int), ssd_buffer_ptr (int).
        Returns empty dict if the key does not exist.
        """
        r = self._client()
        key = f"meta:{int(node_id)}"
        data = r.hgetall(key)
        if not data:
            return {}
        out: Dict[str, Union[int, str]] = {}
        nid = data.get("node_id")
        out["node_id"] = int(nid) if nid is not None and nid != "" else int(node_id)
        out["addr"] = data.get("addr", "")
        out["zmq_addr"] = data.get("zmq_addr", "")
        cb = data.get("cpu_buffer_ptr")
        sb = data.get("ssd_buffer_ptr")
        out["cpu_buffer_ptr"] = int(cb) if cb is not None and cb != "" else 0
        out["ssd_buffer_ptr"] = int(sb) if sb is not None and sb != "" else 0
        return out

    def unregist_node_meta(self, node_id: int) -> bool:
        """Unregister node meta by node_id. Returns True if deleted."""
        r = self._client()
        key = f"meta:{int(node_id)}"
        return bool(r.delete(key))


    def set_node_id(self, node_id: int):
        self._node_id = int(node_id)

    def load_pcfs_file_nodeids(self) -> Dict[int, List[int]]:
        """Load all PCFS file node IDs grouped by node id from Redis.

        - Uses SCAN instead of KEYS to avoid blocking Redis server
        - Scans keys matching pattern "pcfs:*" (each is a list for a node's file node IDs)
        - For each key, fetches the list via LRANGE and converts elements to ints
        - Returns dict: { node_id: [file_nodeid, ...], ... }
        """
        r = self._client()
        result: Dict[int, List[int]] = {}
        try:
            # Use SCAN instead of KEYS to avoid blocking
            cursor = 0
            while True:
                cursor, keys = r.scan(cursor=cursor, match="pcfs:*", count=100)
                for key in keys:
                    try:
                        if not isinstance(key, str):
                            key = str(key)
                        if not key.startswith("pcfs:"):
                            continue
                        nid_part = key.split(":", 1)[1]
                        node_id = int(nid_part)
                    except Exception:
                        continue
                    try:
                        values = r.lrange(key, 0, -1)
                        file_nodeids = [int(v) for v in values]
                    except Exception:
                        file_nodeids = []
                    result[node_id] = file_nodeids
                
                if cursor == 0:
                    break
        except Exception:
            return result
        return result
