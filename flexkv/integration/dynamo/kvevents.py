from __future__ import annotations

import asyncio
import time
from typing import Iterable, List, Optional

import msgpack

try:
    import zmq
    import zmq.asyncio
except ImportError:
    zmq = None
    zmq_asyncio = None

try:
    from nats.aio.client import Client as NATS
    from nats.js.api import StreamConfig
except ImportError:
    NATS = None


class KVEventPublisher:
    def __init__(
        self,
        mode: str = 'zmq',
        zmq_endpoint: str = 'tcp://127.0.0.1:5555',
        zmq_topic: str = 'kv_cache_events',
        nats_servers: str = 'nats://127.0.0.1:4222',
        nats_subject: str = 'kv.cache.events',
        dp_rank: int = 0,
        pod_name: Optional[str] = None,
        lazy_connect: bool = True,
    ) -> None:
        self.mode = mode.lower()
        self.dp_rank = dp_rank
        self.pod_name = pod_name
        self.nats_servers = nats_servers
        self.nats_subject = nats_subject
        self.zmq_endpoint = zmq_endpoint
        self.zmq_topic = zmq_topic.encode('utf-8')
        self._zmq_socket = None
        self._nats_client: Optional[NATS] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lazy_connect = lazy_connect
        self._sequence = 0

        if self.mode == 'zmq':
            self._setup_zmq()
        elif self.mode == 'nats':
            # If not lazy connecting, establish NATS connection immediately.
            if not self._lazy_connect:
                # Acquire the global event loop for NATS operations
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                self._loop.run_until_complete(self._setup_nats())
        else:
            raise ValueError(
                f'Unsupported mode "{self.mode}". Valid options are "zmq" or "nats".'
            )

    def _setup_zmq(self) -> None:
        if zmq is None:
            raise RuntimeError(
                'pyzmq is required for ZMQ publishing but is not installed. '
                'Please install it via "pip install pyzmq".'
            )
        context = zmq.Context.instance()
        socket = context.socket(zmq.PUB)
        socket.bind(self.zmq_endpoint)
        self._zmq_socket = socket

    async def _setup_nats(self) -> None:
        if NATS is None:
            raise RuntimeError(
                'nats-py is required for NATS publishing but is not installed. '
                'Please install it via "pip install nats-py".'
            )
        nc = NATS()
        servers = [s.strip() for s in self.nats_servers.split(',') if s.strip()]
        await nc.connect(servers=servers)
        self._nats_client = nc

    def _create_event(
        self,
        event_type: str,
        event_id: int,
        block_hashes: Iterable[int],
        token_ids: Optional[Iterable[int]] = None,
        lora_id: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> dict:
        payload: dict = {
            'event_id': event_id,
            'event_type': event_type,
            'dp_rank': self.dp_rank,
            'timestamp': time.time(),
            'data': {
                'block_hashes': list(block_hashes),
            },
        }
        if token_ids is not None and event_type == 'BlockStored':
            payload['data']['token_ids'] = list(token_ids)
        if model_name is not None:
            payload['data']['model_name'] = model_name
        if lora_id is not None:
            payload['data']['lora_id'] = lora_id
        if self.pod_name is not None:
            payload['data']['pod_name'] = self.pod_name
        return payload

    def publish_stored(
        self,
        event_id: int,
        block_hashes: Iterable[int],
        token_ids: Iterable[int],
        lora_id: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> None:
        event = self._create_event(
            event_type='BlockStored',
            event_id=event_id,
            block_hashes=block_hashes,
            token_ids=token_ids,
            lora_id=lora_id,
            model_name=model_name,
        )
        self._publish(event)

    def publish_removed(
        self,
        event_id: int,
        block_hashes: Iterable[int],
        lora_id: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> None:
        event = self._create_event(
            event_type='BlockRemoved',
            event_id=event_id,
            block_hashes=block_hashes,
            token_ids=None,
            lora_id=lora_id,
            model_name=model_name,
        )
        self._publish(event)

    def _publish(self, event: dict) -> None:
        if self.mode == 'zmq':
            self._publish_zmq(event)
        elif self.mode == 'nats':
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(self._loop)
                self._loop.run_until_complete(self._publish_nats(event))
            finally:
                asyncio.set_event_loop(None)
        else:
            raise RuntimeError(f'Unknown mode "{self.mode}".')

    def _publish_zmq(self, event: dict) -> None:
        if self._zmq_socket is None:
            self._setup_zmq()

        self._sequence += 1
        batch = {
            'timestamp': event['timestamp'],
            'events': [event],
            'dp_rank': self.dp_rank,
        }
        payload = msgpack.packb(batch, use_bin_type=True)
        # ZMQ multipart: [topic, sequence_bytes, payload]
        sequence_bytes = self._sequence.to_bytes(8, byteorder='big', signed=False)
        self._zmq_socket.send_multipart([
            self.zmq_topic,
            sequence_bytes,
            payload,
        ])

    async def _publish_nats(self, event: dict) -> None:
        if self._nats_client is None:
            await self._setup_nats()

        assert self._nats_client is not None
        nc = self._nats_client

        payload = msgpack.packb(event, use_bin_type=True)
        await nc.publish(self.nats_subject, payload)

    def shutdown(self) -> None:
        if self._zmq_socket is not None:
            try:
                self._zmq_socket.close(0)
            except Exception:
                pass
            self._zmq_socket = None

        if self._nats_client is not None:
            try:
                if self._loop is None:
                    # Should not happen
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(self._nats_client.flush())
                    loop.run_until_complete(self._nats_client.drain())
                else:
                    asyncio.set_event_loop(self._loop)
                    self._loop.run_until_complete(self._nats_client.flush())
                    self._loop.run_until_complete(self._nats_client.drain())
            except Exception:
                pass
            finally:
                self._nats_client = None
        # Stop the event loop if we created one
        if self._loop is not None:
            try:
                self._loop.stop()
                self._loop.close()
            except Exception:
                pass
            finally:
                self._loop = None
