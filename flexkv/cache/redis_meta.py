from __future__ import annotations

from typing import Iterable, List, Tuple
from dataclasses import dataclass
from enum import IntEnum
from uuid import uuid1
try:  # redis-py
    import redis as _redis
except Exception:  # pragma: no cover
    _redis = None  # type: ignore

try:
    from c_ext import RedisMetaChannel as _CRedisMetaChannel, BlockMeta as _CBlockMeta
except Exception as e:  # pragma: no cover
    _CRedisMetaChannel = None  # type: ignore
    _CBlockMeta = None  # type: ignore


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
        return BlockMeta(
            ph=int(cm.ph),
            pb=int(cm.pb),
            nid=int(cm.nid),
            hash=int(cm.hash),
            lt=int(cm.lt),
            state=NodeState(int(cm.state))
        )


class RedisMetaChannel:
    def __init__(self, host: str, port: int, node_id: int, local_ip: str, blocks_key: str = "blocks") -> None:
        if _CRedisMetaChannel is None:
            raise ImportError("c_ext.RedisMetaChannel is not available")
        self._c = _CRedisMetaChannel(host, int(port), int(node_id), str(local_ip), str(blocks_key))

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

    def publish_one(self, meta: BlockMeta) -> None:
        self._c.publish_one(meta.to_c())

    def publish_batch(self, metas: Iterable[BlockMeta], batch_size: int = 100) -> None:
        cms = [m.to_c() for m in metas]
        self._c.publish_batch(cms, int(batch_size))

    def load(self, max_items: int) -> List[BlockMeta]:
        cms = self._c.load(int(max_items))
        return [BlockMeta.from_c(cm) for cm in cms]

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

    def update_block_state_batch(self, node_id: int, hashes: Iterable[int], state: int, batch_size: int = 200) -> None:
        self._c.update_block_state_batch(int(node_id), list(int(h) for h in hashes), int(state), int(batch_size))

    def delete_blockmeta_batch(self, node_id: int, hashes: Iterable[int], batch_size: int = 200) -> None:
        self._c.delete_blockmeta_batch(int(node_id), list(int(h) for h in hashes), int(batch_size))


class RedisMeta:
    def __init__(self, host: str, port: int, password: str | None = None, local_ip: str = "127.0.0.1", decode_responses: bool = True) -> None:
        if _redis is None:  # pragma: no cover
            raise ImportError("redis-py is required: pip install redis")
        self.host = host
        self.port = int(port)
        self.local_ip = str(local_ip)
        self._uuid = str(uuid1())
        self.db = 0
        self.password = password
        self.decode_responses = bool(decode_responses)
        self._node_id: int | None = None

    def _client(self):
        return _redis.Redis(host=self.host, port=self.port, db=self.db, password=self.password, decode_responses=self.decode_responses)

    def init_meta(self) -> int:
        r = self._client()
        node_id = int(r.incr("global:node_id"))
        r.hset(f"node:{node_id}", mapping={"ip": self.local_ip, "uuid": self._uuid})
        self._node_id = node_id
        return node_id

    def get_node_id(self) -> int:
        if self._node_id is None:
            raise RuntimeError("node_id is not registered yet. Call init_meta() first.")
        return int(self._node_id)

    def get_redis_meta_channel(self, blocks_key: str = "blocks") -> "RedisMetaChannel":
        nid = self.get_node_id()
        return RedisMetaChannel(self.host, int(self.port), int(nid), self.local_ip, str(blocks_key))

    def unregister_node(self, node_id: int | None = None) -> None:
        r = self._client()
        nid = int(node_id) if node_id is not None else (self._node_id if self._node_id is not None else -1)
        if nid >= 0:
            r.delete(f"node:{nid}")
        self._node_id = None

    def get_uuid(self) -> str:
        return self._uuid

    def add_node_ids(self, node_ids: Iterable[int | str]) -> int:
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

    def unregist_buffer(self, buffer_ptr: int | str) -> bool:
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

    def regist_node_meta(self, node_id: int, addr: str, cpu_buffer_ptr: int, ssd_buffer_ptr: int) -> None:
        """Register node meta information as a Redis hash.

        Key: meta:<node_id>
        Fields: node_id (int), addr (str), cpu_buffer_ptr (int), ssd_buffer_ptr (int)
        """
        r = self._client()
        key = f"meta:{int(node_id)}"
        r.hset(key, mapping={
            "node_id": int(node_id),
            "addr": str(addr),
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
        out: dict[str, int | str] = {}
        nid = data.get("node_id")
        out["node_id"] = int(nid) if nid is not None and nid != "" else int(node_id)
        out["addr"] = data.get("addr", "")
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


    def load_pcfs_file_nodeids(self) -> dict[int, list[int]]:
        """Load all PCFS file node IDs grouped by node id from Redis.

        - Scans keys matching pattern "pcfs:*" (each is a list for a node's file node IDs)
        - For each key, fetches the list via LRANGE and converts elements to ints
        - Returns dict: { node_id: [file_nodeid, ...], ... }
        """
        r = self._client()
        result: dict[int, list[int]] = {}
        try:
            keys = r.keys("pcfs:*")
        except Exception:
            return result
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
        return result

