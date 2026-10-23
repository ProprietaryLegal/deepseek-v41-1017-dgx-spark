# SPDX-License-Identifier: Apache-2.0
"""RemoteDraftClient against the reference worker over the TCP stub (127.0.0.1)."""

import time

import numpy as np
import pytest

from remote_dspark import wire
from remote_dspark.client import RemoteDraftClient
from remote_dspark.config import RemoteConfig
from remote_dspark.mock_worker import TcpStubServer, WorkerCore

ID = bytes(range(32))
H = 4
RB = 3 * H * 2


def hello(**kw):
    base = dict(hidden=H, row_bytes=RB, drafter_identity=ID)
    base.update(kw)
    return wire.Hello(**base)


def cfg_for(port, **kw):
    env = {"VLLM_DSPARK_REMOTE": f"tcp://127.0.0.1:{port}",
           "VLLM_DSPARK_REMOTE_IDENTITY": ID.hex(),
           "VLLM_DSPARK_REMOTE_TIMEOUT_MS": "200",
           "VLLM_DSPARK_REMOTE_BACKOFF_MS": "5",
           "VLLM_DSPARK_REMOTE_BACKOFF_MAX_MS": "20",
           "VLLM_DSPARK_REMOTE_CONNECT_TIMEOUT_MS": "500"}
    env.update(kw)
    return RemoteConfig.from_env(env)


@pytest.fixture
def server():
    core = WorkerCore(hello())
    srv = TcpStubServer(core, "127.0.0.1", 0).start()
    yield srv
    srv.stop()


def msg_with_rows(n_req, rows):
    buf = bytearray(wire.APP_HEADER_SIZE + wire.rows_offset_for(n_req) + rows.nbytes)
    off = wire.APP_HEADER_SIZE + wire.rows_offset_for(n_req)
    buf[off:off + rows.nbytes] = rows.tobytes()
    return buf


def wait_ready(cli, timeout=3.0):
    t = time.monotonic() + timeout
    while not cli.ready and time.monotonic() < t:
        time.sleep(0.005)
    return cli.ready


def test_connect_hello_step(server):
    cli = RemoteDraftClient(cfg_for(server.port), hello(), background=False)
    assert cli.connect_now() and cli.ready and cli.epoch == 1
    rows = np.ones((2, RB), np.uint8)
    e = [wire.ReqEntry(0, 1, 0, 2, 3, 2, flags=wire.RF_NEED_DRAFT | wire.RF_NEW_GEN)]
    out = cli.step(msg_with_rows(1, rows), e, 2, step_id=1)
    assert out.ok and out.reply.status[0] == wire.RS_OK and out.rtt_us > 0
    st, body = cli.simple_request(wire.OP_PING)
    assert st == wire.ST_OK
    cli.close()


def test_identity_mismatch_disables(server):
    cli = RemoteDraftClient(cfg_for(server.port), hello(drafter_identity=bytes(32)),
                            background=False)
    assert not cli.connect_now()
    assert cli.disabled and "IDENTITY_MISMATCH" in cli.disabled_reason
    out = cli.step(bytearray(4096), [wire.ReqEntry(0, 1, 0, 1, 1, 1)], 1, step_id=1)
    assert not out.ok and out.reason == "disabled"


def test_timeout_then_background_reconnect(server):
    cli = RemoteDraftClient(cfg_for(server.port), hello(), background=True)
    assert wait_ready(cli)
    server.core.inject.delay_ms = 400               # > 200 ms deadline
    e = [wire.ReqEntry(0, 1, 0, 1, 3, 1, flags=wire.RF_NEED_DRAFT)]
    out = cli.step(msg_with_rows(1, np.zeros((1, RB), np.uint8)), e, 1, step_id=1)
    assert not out.ok and out.reason == "timeout" and out.unknown_apply
    assert not cli.ready
    server.core.inject.delay_ms = 0
    assert wait_ready(cli)
    assert cli.epoch == 2
    out = cli.step(msg_with_rows(1, np.zeros((1, RB), np.uint8)), e, 1, step_id=2)
    assert out.ok
    cli.close()


def test_worker_restart_no_session_recovers(server):
    cli = RemoteDraftClient(cfg_for(server.port), hello(), background=True)
    assert wait_ready(cli)
    server.core = WorkerCore(hello())               # worker lost its state
    e = [wire.ReqEntry(0, 1, 0, 1, 3, 1, flags=wire.RF_NEED_DRAFT)]
    out = cli.step(msg_with_rows(1, np.zeros((1, RB), np.uint8)), e, 1, step_id=1)
    assert not out.ok and out.reason == "frame_no_session" and not out.unknown_apply
    assert wait_ready(cli) and cli.epoch == 2
    out = cli.step(msg_with_rows(1, np.zeros((1, RB), np.uint8)), e, 1, step_id=2)
    assert out.ok
    cli.close()


def test_frame_error_keeps_connection(server):
    cli = RemoteDraftClient(cfg_for(server.port), hello(), background=False)
    assert cli.connect_now()
    server.core.inject.frame_status_once = wire.ST_INTERNAL
    e = [wire.ReqEntry(0, 1, 0, 1, 3, 1, flags=wire.RF_NEED_DRAFT)]
    out = cli.step(msg_with_rows(1, np.zeros((1, RB), np.uint8)), e, 1, step_id=1)
    assert not out.ok and out.reason == "frame_internal" and cli.ready
    out = cli.step(msg_with_rows(1, np.zeros((1, RB), np.uint8)), e, 1, step_id=2)
    assert out.ok
    cli.close()


def test_dropped_connection_is_transport_error(server):
    cli = RemoteDraftClient(cfg_for(server.port), hello(), background=False)
    assert cli.connect_now()
    server.core.inject.drop_next_step = True
    e = [wire.ReqEntry(0, 1, 0, 1, 3, 1, flags=wire.RF_NEED_DRAFT)]
    out = cli.step(msg_with_rows(1, np.zeros((1, RB), np.uint8)), e, 1, step_id=1)
    assert not out.ok and out.reason == "transport_error" and out.unknown_apply
    assert not cli.ready


def test_unreachable_worker_never_blocks_engine():
    cli = RemoteDraftClient(cfg_for(1), hello(), background=True)   # port 1: refused
    t0 = time.perf_counter()
    out = cli.step(bytearray(4096), [wire.ReqEntry(0, 1, 0, 1, 1, 1)], 1, step_id=1)
    assert not out.ok and out.reason == "not_ready"
    assert time.perf_counter() - t0 < 0.05
    time.sleep(0.05)
    assert cli.stats.snapshot()["counters"].get("connect_failures", 0) >= 1
    cli.close()
