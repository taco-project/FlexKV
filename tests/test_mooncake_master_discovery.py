"""Master discovery for check_server; etcd and the master are mocked."""

import base64

import pytest

import flexkv.external.mooncake_store_utils as mc
from flexkv.external.mooncake_store_utils import MooncakeStoreClient, MooncakeStoreConfig

pytestmark = pytest.mark.unit

DEFAULT_KEY = "mooncake-store/mooncake_cluster/master_view"


def client(master_addr, **kwargs):
    c = MooncakeStoreClient.__new__(MooncakeStoreClient)
    c._config = MooncakeStoreConfig(master_addr=master_addr, **kwargs)
    return c


class FakeResponse:
    def __init__(self, body=None, text="", error=None):
        self._body = body or {}
        self.text = text
        self._error = error

    def raise_for_status(self):
        if self._error:
            raise self._error

    def json(self):
        return self._body


def etcd_value(value):
    return FakeResponse({"kvs": [{"key": "k", "value": base64.b64encode(value.encode()).decode()}]})


@pytest.fixture(autouse=True)
def _no_cluster_env(monkeypatch):
    monkeypatch.delenv("MC_STORE_CLUSTER_ID", raising=False)


@pytest.fixture
def etcd(monkeypatch):
    """Route etcd range requests to per-endpoint handlers and record them."""
    calls, handlers = [], {}

    def post(url, json=None, timeout=None):
        endpoint = url[: -len("/v3/kv/range")]
        calls.append((endpoint, base64.b64decode(json["key"]).decode()))
        result = handlers[endpoint]() if callable(handlers[endpoint]) else handlers[endpoint]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(mc.requests, "post", post)
    return calls, handlers


@pytest.mark.parametrize(
    "master_addr,expected",
    [
        ("etcd://10.0.0.1:2379;10.0.0.2:2379", ["http://10.0.0.1:2379", "http://10.0.0.2:2379"]),
        ("etcd://10.0.0.1:2379,10.0.0.2:2379", ["http://10.0.0.1:2379", "http://10.0.0.2:2379"]),
        ("etcd://a:2379; b:2379 ,c:2379;", ["http://a:2379", "http://b:2379", "http://c:2379"]),
        ("etcd://h1", ["http://h1:2379"]),
        ("etcd://[::1]:2379", ["http://[::1]:2379"]),
        ("etcd://[::1]", ["http://[::1]:2379"]),
        ("etcd://https://h1:2380/", ["https://h1:2380"]),
    ],
)
def test_etcd_endpoint_parsing(master_addr, expected):
    assert client(master_addr)._etcd_endpoints() == expected


def test_direct_master_skips_etcd(etcd):
    calls, _ = etcd
    c = client("10.0.0.1:50051")
    assert c._etcd_endpoints() == []
    assert c._resolve_master_host() == "10.0.0.1"
    assert calls == []


def test_failover_to_next_endpoint(etcd):
    calls, handlers = etcd
    handlers["http://10.0.0.1:2379"] = ConnectionError("down")
    handlers["http://10.0.0.2:2379"] = etcd_value("10.9.9.9:50051")
    assert client("etcd://10.0.0.1:2379;10.0.0.2:2379")._resolve_master_host() == "10.9.9.9"
    assert calls == [("http://10.0.0.1:2379", DEFAULT_KEY), ("http://10.0.0.2:2379", DEFAULT_KEY)]


def test_http_error_is_not_a_leader(etcd):
    _, handlers = etcd
    handlers["http://h:2379"] = FakeResponse(error=RuntimeError("401"))
    assert client("etcd://h:2379")._resolve_master_host() is None


def test_missing_key_means_no_leader(etcd):
    _, handlers = etcd
    handlers["http://h:2379"] = FakeResponse({})
    assert client("etcd://h:2379")._resolve_master_host() is None


def test_omitted_value_field_returns_none_without_error(etcd):
    # The gRPC-gateway drops empty fields; this must not raise KeyError.
    _, handlers = etcd
    handlers["http://h:2379"] = FakeResponse({"kvs": [{"key": "k"}]})
    assert mc._query_etcd_master_view("http://h:2379", DEFAULT_KEY) is None


def test_ipv6_leader_address(etcd):
    _, handlers = etcd
    handlers["http://h:2379"] = etcd_value("[fd00::1]:50051")
    assert client("etcd://h:2379")._resolve_master_host() == "[fd00::1]"


@pytest.mark.parametrize(
    "env,key",
    [
        (None, DEFAULT_KEY),
        ("", DEFAULT_KEY),
        ("bar", "mooncake-store/bar/master_view"),
        ("bar/", "mooncake-store/bar/master_view"),
    ],
)
def test_master_view_key_follows_sdk_cluster_id(etcd, monkeypatch, env, key):
    calls, handlers = etcd
    if env is not None:
        monkeypatch.setenv("MC_STORE_CLUSTER_ID", env)
    handlers["http://h:2379"] = FakeResponse({})
    client("etcd://h:2379")._resolve_master_host()
    assert calls == [("http://h:2379", key)]


def test_config_cluster_id_is_ignored(tmp_path, monkeypatch):
    # The SDK never reads cluster_id from the JSON config, so neither do we.
    cfg = tmp_path / "mooncake.json"
    cfg.write_text(
        '{"master_addr": "etcd://h:2379", "cluster_id": "foo", "metadata_server": "P2PHANDSHAKE",'
        ' "protocol": "tcp", "device_name": "", "local_hostname": "127.0.0.1",'
        ' "global_segment_size": 1, "enable_ssd_offload": false, "ssd_offload_path": null,'
        ' "master_metrics_port": 9003}'
    )
    monkeypatch.setenv("FLEXKV_MOONCAKE_STORE_CONFIG_PATH", str(cfg))
    config = MooncakeStoreConfig.from_file(object())
    assert not hasattr(config, "cluster_id")
    assert MooncakeStoreClient._cluster_id() == "mooncake_cluster"


def test_check_server_waits_for_election_and_follows_leader_change(etcd, monkeypatch):
    _, handlers = etcd
    views = iter([FakeResponse({}), etcd_value("10.0.0.5:50051"), etcd_value("10.0.0.6:50051")])
    handlers["http://h:2379"] = lambda: next(views)
    probed = []

    def get(url, timeout=None):
        probed.append(url)
        if "10.0.0.5" in url:
            raise ConnectionError("old leader gone")
        return FakeResponse(text="segment")

    monkeypatch.setattr(mc.requests, "get", get)
    monkeypatch.setattr(mc.time, "sleep", lambda _: None)
    client("etcd://h:2379", global_segment_size=0).check_server()
    assert probed == [
        "http://10.0.0.5:9003/get_all_segments",
        "http://10.0.0.6:9003/get_all_segments",
    ]


def test_check_server_times_out(etcd, monkeypatch):
    _, handlers = etcd
    handlers["http://h:2379"] = FakeResponse({})
    monkeypatch.setattr(mc, "SETUP_TIMEOUT", 0)
    with pytest.raises(ValueError, match="timeout"):
        client("etcd://h:2379").check_server()
