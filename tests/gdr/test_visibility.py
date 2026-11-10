#!/usr/bin/env python3
"""Cross-process H2D visibility gate: a consuming kernel in the allocation owner.

``test_direct_ipc.py`` already moves real bytes between two processes, but it
checks them with ``tensor.cpu()`` from the *worker* -- the process that just
issued the RDMA. Both halves of that are too weak to license dropping the XDR
read-back flush, and ``docs/gdr/README.md`` says so: a copy-engine readback need
not share a consuming kernel's visibility path, and the writer's own view says
nothing about what the owner's SMs see.

So this test puts the roles where production has them. The parent is the FlexKV
worker: it imports the owners' KV over IPC and drives layerwise H2D over RDMA.
Each TP rank is its own allocation-owner process, exactly as SGLang runs them:
its own CUDA context, its own eventfds, its own interpreter. It waits on the
layer's fd and launches the comparison itself, with no queue handoff and no
sibling thread in the way.

That per-rank process split is load-bearing, not tidiness. With every rank in
one process the consumers all become runnable on the same burst of eventfd
posts and then serialize on the GIL, which widens the notification-to-launch
window this test exists to keep narrow -- and a wide window lets writes become
visible on their own, which is exactly the bug it is meant to catch.

The comparison is ``torch.ne``, and that choice is the mechanism too: a
comparison is arithmetic, so no copy engine can serve it, which makes it an SM
kernel that reads every byte of the layer from the owner's own context. A
same-dtype ``copy_`` would likely be a DMA -- the one path whose visibility was
never in doubt -- and a gather is not portable here, since XDNN has no
uint8/int64 ``index_select``.

Three further properties are what make this a visibility test rather than a
data test:

* Nothing sits between ``eventfd_read`` and the launch -- no allocation, no
  view construction, no synchronization. Any of those could establish
  visibility on their own and hide the bug.
* Each round writes a **new marker**, and the owner pre-reads the same blocks
  before the round. A read that misses the flush therefore sees the previous
  round's marker, which the report names as stale rather than as noise.
* The verdict is captured by the kernel; the *explanation* is read afterwards.
  So a byte the kernel called wrong but that reads correct a moment later is
  reported for what it is -- a visibility failure -- and not confused with
  corruption, which reads wrong both times.

Odd block IDs are never a transfer target. They are compared too, against zero,
so an out-of-range write is a named failure rather than silent damage.

Run it on the device, with the same environment as the IPC smoke test:

    FLEXKV_LOG_LEVEL=WARNING FLEXKV_TRANSFER_TRACE=0 \
    python tests/gdr/test_visibility.py --tp 8 --qps 4 --rounds 200 \
      --notify polling --expect-flush --nics "$NICS"
"""
import argparse
import collections
import ctypes
import gc
import multiprocessing as mp
import os
import select
import signal
import socket
import sys
import threading
import time
from multiprocessing.reduction import recvfds, sendfds
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Before torch, because XMLIR reads these as its runtime initializes. Without
# them a device allocation gets no peer-memory host mapping, and every owner
# then dies deep inside GdrOwner on "XDR mapping failed" -- eight tracebacks for
# one unset variable. setdefault, so a caller who chose otherwise still wins,
# covered before their own torch import rather than only by inheritance.
os.environ.setdefault("BKCL_ENABLE_XDR", "1")
os.environ.setdefault("CUDA_ENABLE_P2P_NO_UVA", "1")
import torch
from flexkv.common.gdr_config import parse_gdr_config
from flexkv.common.memory_handle import (
    TensorSharedHandle, close_all_cuda_ipc_handles, _IS_XMLIR,
)
from flexkv.transfer.gdr_owner import GdrOwner, attach_remote_targets, close_leases
from flexkv.transfer.region_batch import RegionSpec, build_region_batch, make_requests

CAPACITY = 16                       # blocks per rank
TOUCHED = list(range(0, CAPACITY, 2))   # GPU blocks the worker writes
IDLE = list(range(1, CAPACITY, 2))      # GPU blocks nothing may ever write
HOST_IDS = list(range(1, CAPACITY, 2))  # host blocks read, deliberately != TOUCHED
UNWRITTEN = 0                       # value every untouched byte keeps all run


def marker_of(round_id):
    """Differs from its predecessor and from UNWRITTEN, so a read that missed
    the flush is identifiable as stale instead of merely wrong."""
    return 1 + round_id % 251


def explain(wrong, later, marker, stale):
    """Turn one failed comparison into a sentence.

    ``wrong`` is the mask the consuming kernel produced at notification time;
    ``later`` is what those same bytes read as once the dust settled. The gap
    between the two carries the diagnosis. Bytes the kernel rejected that now
    hold this round's marker arrived late -- that is a visibility failure, and
    it is the exact signature the XDR read-back exists to prevent. Bytes that
    are wrong in both reads are corruption, a different bug.
    """
    block, byte = (int(v) for v in wrong.nonzero()[0])
    seen = sorted(int(v) for v in torch.unique(later[wrong])[:8])
    count = int(wrong.sum())
    if block not in TOUCHED:
        return (f"out-of-range write: gpu_block={block} byte={byte} must stay "
                f"{UNWRITTEN}, reads {seen}, mismatched={count}")
    kind = ("VISIBILITY FAILURE: the kernel missed bytes that are present now"
            if seen == [marker] else
            "stale: still the previous round's marker" if seen == [stale] else
            "never written" if seen == [UNWRITTEN] else "torn or foreign")
    return (f"gpu_block={block} byte={byte} expected={marker} "
            f"later_reads={seen} mismatched={count} ({kind})")


def self_check():
    """Run before anything else, on every invocation, on the real machine: a
    green gate is only evidence if the detector can go red on the same box."""
    shape, marker, stale = (CAPACITY, 8), marker_of(5), marker_of(4)
    for fill, expected in ((marker, "VISIBILITY FAILURE"), (stale, "stale"),
                           (UNWRITTEN, "never written"), (marker + 1, "torn")):
        wrong = torch.zeros(shape, dtype=torch.bool)
        wrong[TOUCHED[1], 3] = True
        report = explain(wrong, torch.full(shape, fill, dtype=torch.uint8),
                         marker, stale)
        assert expected in report, (fill, report)
        assert f"gpu_block={TOUCHED[1]} byte=3" in report, report
    wrong = torch.zeros(shape, dtype=torch.bool)
    wrong[IDLE[0], 5] = True
    report = explain(wrong, torch.full(shape, marker, dtype=torch.uint8),
                     marker, stale)
    assert "out-of-range" in report, report


def describe_gaps(gaps):
    """Summarize the notification-to-launch distribution, one entry per check.

    This is the gate's self-diagnostic, and it has to be a distribution rather
    than a worst case. Sensitivity is per sample: a check that launched a few
    microseconds after the eventfd is a sharp trial, one that launched half a
    millisecond later is a blunt one, and the run's power comes from the sharp
    end. Reporting only the maximum hid that.
    """
    if not gaps:
        return "n/a"
    ordered = sorted(gaps)
    pick = lambda q: ordered[min(len(ordered) - 1, int(len(ordered) * q))] / 1e3
    return (f"n={len(ordered)} min={ordered[0]/1e3:.1f}us p50={pick(0.5):.1f}us "
            f"p95={pick(0.95):.1f}us max={ordered[-1]/1e3:.1f}us")


def describe_layer_ready(told, cq_last):
    """Describe arrivals without treating consumer slowness as overlap gain.

    Each sample is the last owner's observation for that layer in one round.
    These are CPU timestamps, not GPU execution or compute demand timestamps.
    Data arriving before it is needed is useful even with no temporal overlap.
    """
    rows = []
    for layer, samples in enumerate(told):
        if not samples:
            continue
        observed = sorted(samples)[len(samples) // 2] / 1e3
        text = f"L{layer}={observed:.0f}"
        if layer < len(cq_last) and len(cq_last[layer]) == len(samples):
            # Subtract paired rounds before taking a median.
            delays = sorted(t - c for t, c in zip(samples, cq_last[layer]))
            text += f"(completion_to_observation={delays[len(delays)//2]/1e3:.0f})"
        rows.append(text)
    if not rows:
        return "n/a"
    return ("median us after submit: " + " ".join(rows)
            + "; notification distribution only; not a compute-overlap score")


def validate_notify_mode(notify, is_xmlir):
    """Reject unsupported callbacks before allocating or spawning owners."""
    if is_xmlir and notify == "hostfunc":
        raise ValueError("XMLIR on some special machine types and scenarios does not support hostfunc; use --notify polling")


def describe_laggards(counts, tp):
    """Which rank was the last to drain a layer, and how often.

    A layer is not announced until its slowest rank is done, so once the
    notification itself is prompt this is what caps the overlap.

    One rank holding the title is the interesting case and this function cannot
    resolve it, which is worth saying rather than guessing. A NIC on a worse
    PCIe path and a rank thread that happened to land on a busy core both look
    like "r2 is last 92% of the time" from inside a single run, and they call
    for opposite work. Only repeated runs separate them: on some special machine types and scenarios three runs
    named r7, then r1, then r2, each above half, which is placement and not
    topology -- and a verdict that had asserted "machine asymmetry" would have
    sent the effort to the NICs, which turned out to be mapped correctly.
    """
    total = sum(counts.values())
    if not total or tp < 2:
        return "n/a"
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    body = " ".join(f"r{rank}={n / total:.0%}" for rank, n in ordered)
    share = ordered[0][1] / total
    verdict = (f"rank {ordered[0][0]} is last {share / (1 / tp):.1f}x more often "
               f"than an even split. One run cannot say why: re-run and compare "
               f"-- the same rank every time is topology, a different rank every "
               f"time is where its threads landed at startup"
               if share > 2 / tp else
               "no rank is persistently last, so this is jitter: chasing it "
               "means not waiting on every rank, not finding the slow one")
    return f"{body} -> {verdict}"


RANK_HEADER = ("rank", "finished", "step_p50", "step_p95", "step_max",
               "fence_p50")


def describe_ranks(totals, steps, fences):
    """Each rank's own pace, which is what says *how* the slowest one is slow.

    The timeline tables keep a min and a max across ranks per layer, so a rank
    that is behind shows up only as the gap between two columns. That gap has
    two very different causes and the same appearance: a rank that costs more
    on every single layer, and a rank that keeps up but stalls a few times.

    ``step_p50`` against ``step_max`` separates them. Uniformly slower moves
    the median -- the path, the NIC, or memory bandwidth is the suspect, and
    every layer pays. A normal median with a long tail is a rank losing its
    core now and again, where the suspect is placement and the fix is nowhere
    near the fabric.

    ``fence_p50`` is inside a layer rather than between two, so a rank whose
    steps are long while its fences are not is waiting to *start* layers, not
    to finish them.
    """
    usable = [r for r in range(len(totals)) if steps[r] and fences[r]]
    if not usable:
        return "n/a"
    pick = lambda xs, q: sorted(xs)[min(len(xs) - 1, int(len(xs) * q))] / 1e3
    rows = [(r, sorted(totals[r])[len(totals[r]) // 2] / 1e3,
             pick(steps[r], 0.5), pick(steps[r], 0.95), max(steps[r]) / 1e3,
             pick(fences[r], 0.5)) for r in usable]
    width = max(max(len(f"{v:.1f}") for row in rows for v in row[1:]),
                max(len(name) for name in RANK_HEADER[1:]))
    lines = [f"{RANK_HEADER[0]:>5} " + "".join(f"{name:>{width+1}}"
                                               for name in RANK_HEADER[1:])]
    for rank, *values in rows:
        lines.append(f"r{rank:<4} "
                     + "".join(f"{v:>{width+1}.1f}" for v in values))
    slowest = max(rows, key=lambda row: row[1])
    # Against the *fastest* rank, not the median one. The case this has to
    # catch is a graded spread over all of them -- on some special machine types and scenarios the column ran
    # 35.8us to 57.0us with no outlier -- where slowest-over-median is 1.15
    # and reports "no pace problem" about a column that is 1.59x end to end.
    quickest = min(row[2] for row in rows)
    excess = slowest[2] / quickest if quickest else 1.0
    # A tail every rank shares is the host stalling all of them at once, and
    # blaming the rank that happens to finish last for it sends the reader to
    # look at one thread when eight of them stopped together.
    tails = sorted(row[4] for row in rows)
    shared = len(tails) > 1 and tails[0] > 0.7 * tails[-1]
    if excess > 1.25:
        why = (f"pays {excess:.2f}x the fastest rank's per-layer pace on every "
               f"layer -- uniformly slower, so look at its path and bandwidth")
    elif shared:
        why = (f"keeps a normal per-layer pace ({slowest[2]:.1f}us against "
               f"{quickest:.1f}us), and the {tails[-1]:.0f}us tail is on every "
               f"rank ({tails[0]:.0f}us at least) -- the host stalled all of "
               f"them together, so this column says nothing about this rank")
    else:
        why = (f"keeps a normal per-layer pace ({slowest[2]:.1f}us against "
               f"{quickest:.1f}us) with a tail to {slowest[4]:.0f}us that its "
               f"peers do not share -- it stalls rather than crawls, so look "
               f"at where its threads run")
    # Worth saying even when the pace already explains the total, because
    # step_max is the one column a reader will otherwise chase into a rank
    # that was only standing next to the stall.
    if shared and excess > 1.25:
        why += (f". Its {slowest[4]:.0f}us tail is on every rank "
                f"({tails[0]:.0f}us at least), so that column is the host")
    lines.append(f"us; rank {slowest[0]} finished last at {slowest[1]:.0f}us "
                 f"against {min(row[1] for row in rows):.0f}us for the first, "
                 f"and {why}")
    return "\n             ".join(lines)


def describe_drift(windows):
    """Whether the run was measuring one machine the whole way through.

    Every table here is a median over rounds, which takes the rounds to be
    samples of one thing. When a run degrades partway -- another tenant, a
    thermal cap, this process's own growing heap -- the median blends two
    machines and describes neither, while still printing to three digits.

    On some special machine types and scenarios a run opened at 252us to layer 0 and closed at 465us. Read as one
    median it looks like a 90% regression against the run before it; read as
    two halves it is one good run and one bad one, and the question is what
    changed at the halfway point rather than what changed in the code.
    """
    if len(windows) < 8:
        return ""
    half = len(windows) // 2
    median = lambda xs: sorted(xs)[len(xs) // 2] / 1e3
    early, late = median(windows[:half]), median(windows[half:])
    drift = late / early if early else 1.0
    if 0.9 <= drift <= 1.1:
        return (f"stationary: the last {len(windows) - half} rounds are within "
                f"10% of the first {half} ({early:.0f}us vs {late:.0f}us), so "
                f"the medians below describe one machine")
    return (f"NOT STATIONARY: the run {'slowed' if drift > 1 else 'sped up'} "
            f"from {early:.0f}us to {late:.0f}us ({drift:.2f}x) between its "
            f"halves, so every median below blends two machines and none of "
            f"them is comparable to another run. Find what changed first")


HEADER = ("layer", "post_first", "post_last", "cq_first", "cq_last", "fd",
          "notified", "idle")


def render(rows, note):
    """Lay out one of the two timeline tables. Both carry the same columns, so
    a reader who has learned to read one has learned to read the other.

    ``fd`` is when the worker finished writing the layer's eventfds, which
    splits ``idle`` where the blame changes hands: ``fd - cq_last`` is the
    polling thread being woken by the layer's last rank, scheduled, and getting
    through the writes; ``notified - fd`` is the owners taking that long to be
    scheduled after that. Those are different bugs, and without this column they
    are one number.

    ``fd - cq_last`` is also the column to watch for the one way this design
    can come apart. The poller posts the layers in order on one thread, so if
    the writes for a layer cost more than the gap between layers arriving, it
    falls behind and the lag accumulates down the table rather than staying
    flat.
    """
    # Wide enough for the longest header, or the titles drift out of line with
    # the columns they name and the table becomes something to decode.
    digits = max(len(f"{value:.0f}") for row in rows for value in row[1:])
    width = max(digits, max(len(name) for name in HEADER[1:]))
    lines = [f"{HEADER[0]:>5} " + "".join(f"{name:>{width+1}}"
                                          for name in HEADER[1:])]
    for layer, pf, pl, cf, cl, fd, told in rows:
        lines.append(f"L{layer:<4} "
                     + "".join(f"{value:>{width+1}.0f}"
                               for value in (pf, pl, cf, cl, fd, told, told - cl)))
    lines.append(note)
    return "\n             ".join(lines)


def describe_timeline(post_first, post_last, cq_first, cq_last, fd, told,
                      joined, entry_spread):
    """Five timestamps per layer, on one clock, so the overlap question stops
    being an inference and becomes a table.

    Every column is a median over rounds of a *within-round* extreme across
    ranks, which keeps the columns on one basis and their differences
    meaningful. ``post_first``/``post_last`` bracket when the layer's requests
    reached the NIC lanes, ``cq_first``/``cq_last`` when their completions came
    back, and ``told`` is when the last of the layer's consumers woke -- the
    last and not a median, because pairing a median with ``cq_last``, which is
    a maximum, would subtract two different populations and report a wait no
    rank actually had.

    Three questions come out of reading down the columns:

    * ``post_first[L+1]`` against ``cq_last[L]`` is pipelining: whether the NIC
      starts the next layer before the previous one has been fully reaped.
    * ``post_last - post_first`` is rank skew going in and
      ``cq_last - cq_first`` skew coming out. A layer is not readable until its
      slowest rank is done, so this is what caps the overlap once notification
      is prompt -- and having both says whether the skew is already there at
      submission or develops over the transfer, which is the difference between
      looking at the launcher and looking at the fabric.
    * ``told - cq_last`` is how long a layer sat finished before everyone knew,
      and ``fd`` splits it into the worker's half and the owners' half.
    """
    rows = []
    median = lambda xs: sorted(xs)[len(xs) // 2] / 1e3
    for layer, columns in enumerate(zip(post_first, post_last, cq_first,
                                        cq_last, fd, told)):
        if all(columns):
            rows.append((layer, *map(median, columns)))
    if not rows:
        return "n/a"
    mean = lambda f: sum(f(*row[1:]) for row in rows) / len(rows)
    note = (f"us after submit; a layer waits "
            f"{mean(lambda pf, pl, cf, cl, d, n: n - cl):.0f}us on average "
            f"between its slowest rank's completions being collected and its "
            f"consumer being told -- "
            f"{mean(lambda pf, pl, cf, cl, d, n: d - cl):.0f}us of that before "
            f"the eventfd is written and "
            f"{mean(lambda pf, pl, cf, cl, d, n: n - d):.0f}us waiting for the "
            f"owners to be scheduled -- over a "
            f"{rows[-1][4] - rows[0][1]:.0f}us transfer whose ranks differ by "
            f"{mean(lambda pf, pl, cf, cl, d, n: pl - pf):.0f}us going in and "
            f"{mean(lambda pf, pl, cf, cl, d, n: cl - cf):.0f}us coming out, "
            f"having entered the fan-out {median(entry_spread):.0f}us apart; the "
            f"fan-out joined at {median(joined):.0f}us, which no longer floors "
            f"notification -- the poller is armed before the launches -- but "
            f"does bound the launcher, so a layer told after it waited on a "
            f"rank thread rather than on the fabric")
    return render(rows, note)


def describe_round(rounds):
    """One round's own numbers, as a timeline rather than a distribution.

    The table above medians each column independently over every round and
    every rank, which is the right way to read a distribution and the wrong way
    to read a sequence. Differences between columns become differences of
    medians, and ``cq_last`` is a maximum by construction, so those differences
    come out better than any round actually was. Here every number in a row
    came from the same transfer, so ``notified - cq_last`` is a wait that
    really happened and ``posted[L+1] - cq_last[L]`` is a gap the NIC really
    idled through.

    The round is the median one by notification lateness -- not the luckiest,
    not an outlier -- and the spread is reported alongside so a reader can see
    which of those it nearly was.
    """
    usable = [r for r in rounds if r]
    if not usable:
        return "n/a"
    lateness = lambda r: sum(row[6] - row[4] for row in r) / len(r) / 1e3
    ranked = sorted(range(len(usable)), key=lambda i: lateness(usable[i]))
    pick = ranked[len(ranked) // 2]
    rows = [(row[0], *(value / 1e3 for value in row[1:])) for row in usable[pick]]
    note = (f"us after submit, within round {pick} alone -- the median of "
            f"{len(usable)} by notification lateness, which ran from "
            f"{lateness(usable[ranked[0]]):.0f}us at best through "
            f"{lateness(usable[pick]):.0f}us here to "
            f"{lateness(usable[ranked[-1]]):.0f}us at worst")
    return render(rows, note)


def probe_consuming_kernel(device):
    """Check the one device op this gate depends on, before the IPC and GDR
    handshakes, so a vendor gap says so by name instead of surfacing as a
    confusing failure ten seconds later. XDNN's uint8 coverage does have holes
    -- ``index_select`` on uint8 is one -- and this also confirms on-device
    that the comparison really discriminates, which is what ``self_check``
    establishes for the host-side classifier."""
    try:
        same = torch.zeros((2, 4), dtype=torch.uint8, device=device)
        other = torch.ones((2, 4), dtype=torch.uint8, device=device)
        mask = torch.zeros((2, 4), dtype=torch.bool, device=device)
        torch.ne(same, same, out=mask)
        assert not bool(mask.cpu().any()), "torch.ne reported equal bytes unequal"
        torch.ne(same, other, out=mask)
        assert bool(mask.cpu().all()), "torch.ne reported unequal bytes equal"
    except Exception as exc:
        raise RuntimeError(
            "this gate's consuming kernel -- torch.ne(uint8, uint8, out=bool) on "
            f"{device} -- is not usable here: {exc!r}. It has to be an op no copy "
            "engine can serve, so a plain copy_ is not a substitute; pick another "
            "arithmetic op with the same property.") from exc


def send_fds(conn, fds):
    """Hand this rank's eventfds to the worker over the existing pipe.

    Production does this with SCM_RIGHTS too. It has to happen before any
    pickled message: the fds ride one byte of the stream, and a plain read that
    consumed that byte would drop them.
    """
    with socket.fromfd(conn.fileno(), socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sendfds(sock, fds)


def recv_fds(conn, count):
    with socket.fromfd(conn.fileno(), socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        return recvfds(sock, count)


def receive(conn, timeout=300):
    assert conn.poll(timeout), "owner process timeout"
    result = conn.recv()
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(result["error"])
    return result


def start_background_load(rank, stop):
    """SGLang is computing while FlexKV loads. A side stream keeps the device
    and its caches busy without serializing the consumer."""
    def load():
        torch.cuda.set_device(rank)
        stream = torch.cuda.Stream(device=rank)
        # Built on the host and moved: XDNN's device-side RNG coverage is not
        # what this flag is here to exercise.
        a = torch.randn(2048, 2048, dtype=torch.float16).to(f"cuda:{rank}")
        with torch.cuda.stream(stream):
            while not stop.is_set():
                for _ in range(32):
                    a = torch.mm(a, a).clamp_(-1, 1)
                # Enqueue in bursts and then yield, so this thread cannot hold
                # the GIL against the main loop's launches.
                time.sleep(0.002)
    threading.Thread(target=load, daemon=True).start()


def die_with_parent():
    """Ask the kernel to kill this process when the worker goes away.

    Ignoring SIGINT below is only safe if a worker always survives to do the
    shutdown, and a worker can be killed in ways it cannot respond to. Without
    this, such a death leaves eight owners parked in ``select`` for their
    per-layer timeout with nothing left to terminate them -- the interrupt
    appears to do nothing for minutes. PR_SET_PDEATHSIG is Linux-only, which
    this test already is, and the getppid check closes the race where the
    worker died between the fork and this call.
    """
    PR_SET_PDEATHSIG = 1
    try:
        ctypes.CDLL(None, use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL,
                                                0, 0, 0)
    except Exception:
        return
    if os.getppid() == 1:
        os._exit(1)


def owner_process(conn, args, config, rank):
    """One allocation owner: one SGLang TP rank, in its own process.

    There is deliberately no consumer thread. The main loop arms, answers the
    worker, then waits on this rank's fds and launches the comparison itself --
    the shortest path from notification to kernel that Python allows.
    """
    # Ctrl-C reaches every process in the foreground group at once, and an owner
    # that acts on it unwinds its own MRs while the worker still holds QPs
    # against them -- which is how an interrupt turns into eight tracebacks and
    # a "GDR drain acknowledgement failed" rather than an exit. Releasing
    # registrations in the right order is the worker's job, so decline the
    # signal and wait to be told; the worker's teardown closes this pipe, and
    # the recv below raises EOFError when it does.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    die_with_parent()
    owner = None
    stop_load = threading.Event()
    try:
        # Created and handed over first: everything below can fail, and a worker
        # blocked in recvmsg would have no error to report.
        fds = [os.eventfd(0, os.EFD_NONBLOCK | os.EFD_SEMAPHORE | os.EFD_CLOEXEC)
               for _ in range(args.layers)]
        send_fds(conn, fds)

        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
        buffer = torch.zeros((args.layers, args.kv_dim, CAPACITY, args.fragment),
                             dtype=torch.uint8, device=device)
        probe_consuming_kernel(device)

        # Registration order must match ptr_at<SGLANG>: kv major, layer minor.
        registered = [buffer[l, k] for k in range(args.kv_dim)
                      for l in range(args.layers)]
        owner = GdrOwner(registered, config)
        handles = []
        for i, view in enumerate(registered):
            handle = TensorSharedHandle(view, rank, force_direct_ipc=True)
            handle.gdr_target = owner.metadata(i)
            handles.append(handle)
        torch.cuda.synchronize(rank)

        # Hot-path state, allocated once, including the per-layer views: an
        # allocation or even an aten::select between the notification and the
        # launch is a step the real consumer does not take, and any of them
        # could establish visibility by itself and hide the bug.
        views = [[buffer[l, k] for k in range(args.kv_dim)]
                 for l in range(args.layers)]
        expected = torch.zeros((CAPACITY, args.fragment), dtype=torch.uint8,
                               device=device)
        expected_host = torch.zeros((CAPACITY, args.fragment), dtype=torch.uint8)
        masks = [[torch.zeros((CAPACITY, args.fragment), dtype=torch.bool,
                              device=device) for _ in range(args.kv_dim)]
                 for _ in range(args.layers)]

        def prime(marker):
            """Publish this round's expectation, and pull the *previous*
            round's bytes through the caches for exactly the blocks about to be
            overwritten. A cold line would send the consumer to memory and hide
            the staleness this test hunts for.

            The expectation is built on the host and copied down rather than
            filled in place, so the only compute this needs is the same
            ``torch.ne`` the consumer uses -- no second op to be missing.
            """
            expected_host.fill_(UNWRITTEN)
            expected_host[TOUCHED] = marker
            expected.copy_(expected_host)
            for layer in range(args.layers):
                for k in range(args.kv_dim):
                    torch.ne(views[layer][k], expected, out=masks[layer][k])
            # Must land before the worker starts writing, or the prime is itself
            # a racing reader and the test goes flaky for an unrelated reason.
            torch.cuda.synchronize(rank)

        def verify(round_id, marker):
            """Post hoc. The comparison kernels have already run, so nothing
            here can repair a stale read -- only explain one."""
            torch.cuda.synchronize(rank)
            stale = marker_of(round_id - 1) if round_id else UNWRITTEN
            failures = []
            for layer in range(args.layers):
                for k in range(args.kv_dim):
                    wrong = masks[layer][k].cpu()
                    if not bool(wrong.any()):
                        continue
                    # Only now, and only on failure, read the bytes themselves:
                    # what they say *after* the verdict is what separates a late
                    # arrival from real corruption.
                    later = views[layer][k].cpu()
                    failures.append(f"rank={rank} layer={layer} kv={k} "
                                    + explain(wrong, later, marker, stale))
            return failures

        # Bound once, for the hot path's sake, and CLOCK_MONOTONIC rather than
        # perf_counter because these timestamps are subtracted from one taken
        # in *another process*: the monotonic clock is system-wide, while
        # perf_counter's reference point is not guaranteed to be shared.
        clock, mono = time.clock_gettime_ns, time.CLOCK_MONOTONIC

        def consume(round_id, marker):
            failures, gaps, arrivals, waits = [], [], [], []
            try:
                for layer in range(args.layers):
                    fd = fds[layer]
                    wait_started = clock(mono)
                    if not select.select([fd], [], [], args.timeout)[0]:
                        failures.append(f"rank={rank} layer={layer}: no layer-ready "
                                        f"within {args.timeout}s")
                        break
                    os.eventfd_read(fd)
                    woke = clock(mono)
                    for k in range(args.kv_dim):
                        torch.ne(views[layer][k], expected, out=masks[layer][k])
                    gaps.append(clock(mono) - woke)
                    arrivals.append(woke)
                    waits.append(woke - wait_started)
                failures = failures or verify(round_id, marker)
            except Exception as exc:
                # Name the harness bug rather than letting it look like a fence
                # failure in the report. Exception and not BaseException: an
                # interrupt turned into a failure string is an interrupt this
                # loop then ignores, and it would arrive at the worker looking
                # like a visibility bug.
                failures = [f"rank={rank} consumer raised: {exc!r}"]
            return failures, gaps, arrivals, waits

        # Pay this process's first-launch cost here, not inside the window under
        # test: XMLIR charges tens of milliseconds for the first launch.
        torch.ne(views[0][0], expected, out=masks[0][0])
        torch.cuda.synchronize(rank)
        if args.background:
            start_background_load(rank, stop_load)

        conn.send(handles)
        while True:
            command = conn.recv()
            if command[0] == "arm":
                round_id, marker = command[1], command[2]
                prime(marker)
                conn.send("armed")
                # Straight into the wait. Nothing stands between the fd and the
                # kernel but the read itself.
                conn.send(consume(round_id, marker))
            elif command[0] == "stop":
                stop_load.set()
                owner.close()
                owner = None
                conn.send("owner MRs released")
                return
            else:
                raise RuntimeError(command)
    except BaseException as exc:
        stop_load.set()
        try:
            conn.send({"error": repr(exc)})
        except OSError:
            # The worker is already gone, almost always because a sibling hit
            # this same error first and it propagated. Reporting ours down a
            # dead pipe only replaces the real traceback with a BrokenPipeError
            # one, which is how a single unset variable turned into pages of
            # noise that buried its own cause.
            pass
        raise


def worker_process(conns, args, config):
    """The FlexKV worker: imports every owner's KV and writes it over RDMA."""
    fds = []
    for conn in conns:
        fds += recv_fds(conn, args.layers)   # rank major: rank*layers + layer
    handles = [receive(conn) for conn in conns]
    # The thread count is a precondition, not a curiosity: torch sizes its
    # intra-op pools from the core count, so on a 384-core host nine processes
    # bring thousands of spinning threads and every wait below measures them.
    print("[visibility] handle_types=" + ",".join(
        sorted({h.handle_type for rank in handles for h in rank}))
        + f"; worker_threads={len(os.listdir('/proc/self/task'))}"
        + f"; omp={os.environ.get('OMP_NUM_THREADS', 'unset')}", flush=True)
    if not os.environ.get("OMP_NUM_THREADS") and os.cpu_count() > 32:
        # Left unset on this host, every number below measured OpenMP: layer 0
        # was announced at 48% of the window instead of 8%, and the owners' wait
        # for a core was 482us instead of 5us. Cheaper to say so once than to
        # have someone rediscover it from a timeline table.
        print(f"[visibility] WARNING: OMP_NUM_THREADS is unset on a "
              f"{os.cpu_count()}-core host, so torch will size its intra-op "
              f"pools from the core count and every wait reported below will "
              f"measure thousands of spinning threads. Set OMP_NUM_THREADS=1 "
              f"and MKL_NUM_THREADS=1 before trusting any of it.", flush=True)
    tensors = [[h.get_tensor() for h in rank] for rank in handles]

    storage = torch.zeros(args.layers * args.kv_dim * CAPACITY * args.fragment,
                          dtype=torch.uint8, pin_memory=True)
    block = args.layers * args.kv_dim * args.fragment      # BLOCKFIRST host layout
    spec = RegionSpec("kv", storage.data_ptr(), args.fragment, args.kv_dim * args.fragment,
                      block, 0,
                      [t.data_ptr() for rank in tensors for t in rank],
                      args.layers * args.kv_dim,
                      [CAPACITY * args.fragment] * args.tp, [args.fragment] * args.tp,
                      [args.kv_dim * CAPACITY * args.fragment] * args.tp,
                      [args.fragment] * args.tp, args.layers, args.kv_dim, 1)
    group = build_region_batch([spec], list(range(args.tp)), is_blockfirst=True)
    group.configure_gdr(config)
    leases = attach_remote_targets(group, [t for rank in tensors for t in rank],
                                   list(range(args.tp)), config)
    assert len(leases) == args.tp, "direct GDR handshake failed; fallback is forbidden here"
    group.set_layer_eventfds(torch.tensor(fds, dtype=torch.int32), args.tp,
                             args.layers, args.notify)

    gpu_ids = torch.tensor(TOUCHED, dtype=torch.int64).pin_memory()
    cpu_ids = torch.tensor(HOST_IDS, dtype=torch.int64).pin_memory()
    requests = []
    for layer in range(args.layers):
        request = make_requests(1, gpu_ids, cpu_ids, True, region_indices=[0],
                                layer_id=layer, layer_granularity=1)[0]
        request.milestone_layer = layer
        requests.append(request)

    gaps, waits, started = [], [], time.perf_counter()
    # All three are offsets from the same round's submit, on CLOCK_MONOTONIC:
    # the first two come from the worker's own C++ layer boundary, the third
    # from the owner processes. steady_clock is CLOCK_MONOTONIC on Linux, which
    # is what makes the three directly comparable.
    # Every column keeps one within-round extreme across ranks, so that any two
    # of them can be subtracted and the answer is a wait some rank really had.
    columns = [[[] for _ in range(args.layers)] for _ in range(6)]
    post_first, post_last, cq_first, cq_last, fd, told = columns
    joined, entry_spread, stream_free, single, windows = [], [], 0, [], []
    laggards = collections.Counter()
    # Per rank rather than per layer: the columns above are extremes across
    # ranks, so a rank that is behind appears only as the distance between two
    # of them, and never as its own pace. See describe_ranks.
    rank_totals = [[] for _ in range(args.tp)]
    rank_steps = [[] for _ in range(args.tp)]
    rank_fences = [[] for _ in range(args.tp)]
    assert hasattr(group, "layer_timeline_started"), (
        "this extension predates the fan-out entry timestamp; rebuild it, or "
        "the run cannot tell a rank thread that started late from one that was "
        "slow over its first layer")
    group.set_layer_timeline(True)
    try:
        for round_id in range(args.rounds):
            marker = marker_of(round_id)
            storage.fill_(marker)
            for conn in conns:
                conn.send(("arm", round_id, marker))
            for conn in conns:
                assert receive(conn) == "armed"
            # Taken here, in the worker, and subtracted in the owners' clock
            # domain: the owners are already parked in select() by now, so they
            # cannot be handed this beforehand.
            submitted = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            group.submit_layerwise(requests, [], 0)
            ok, error = group.wait_layer_completion(args.timeout)
            assert ok, f"round {round_id}: native layer completion failed: {error}"
            # One row per (layer, rank). A layer is only readable once its
            # slowest rank has drained, so keep that separately rather than
            # letting the median across ranks stand in for it. first/last also
            # feed this round's own table, where the two must come from the
            # same transfer to be subtracted from each other.
            began, ended, first, last, wrote, laggard = {}, {}, {}, {}, {}, {}
            walked = collections.defaultdict(dict)
            for layer, rank, post_ns, reap_ns, told_ns in group.layer_timeline():
                post, reap = post_ns - submitted, reap_ns - submitted
                walked[rank][layer] = (post, reap)
                began[layer] = min(began.get(layer, post), post)
                ended[layer] = max(ended.get(layer, post), post)
                first[layer] = min(first.get(layer, reap), reap)
                if reap > last.get(layer, -1):
                    last[layer], laggard[layer] = reap, rank
                # The layer's, not the rank's: the polling thread posts once
                # per layer and every row of that layer carries the same
                # stamp. Zero only if the extension did not record one.
                if told_ns:
                    wrote[layer] = max(wrote.get(layer, 0), told_ns - submitted)
            for layer in last:
                post_first[layer].append(began[layer])
                post_last[layer].append(ended[layer])
                cq_first[layer].append(first[layer])
                cq_last[layer].append(last[layer])
                fd[layer].append(wrote.get(layer, last[layer]))
            laggards.update(laggard.values())
            for rank, seen in walked.items():
                order = sorted(seen)
                rank_totals[rank].append(seen[order[-1]][1])
                previous = None
                for layer in order:
                    post, reap = seen[layer]
                    rank_fences[rank].append(reap - post)
                    # Boundary to boundary, so it covers this layer's launches
                    # as well as its fence: a rank that is late to *start* a
                    # layer is exactly what the fence column cannot show.
                    if previous is not None:
                        rank_steps[rank].append(reap - previous)
                    previous = reap
            joined.append(group.layer_timeline_joined() - submitted)
            entered = group.layer_timeline_started()
            entry_spread.append(max(entered) - min(entered))
            stream_free += group.layer_stream_free_layers()
            failures = []
            woke_at = {}
            for conn in conns:
                rank_failures, rank_gaps, rank_arrivals, rank_waits = receive(conn)
                failures += rank_failures
                gaps += rank_gaps
                waits += rank_waits
                for layer, woke in enumerate(rank_arrivals):
                    woke_at[layer] = max(woke_at.get(layer, 0), woke - submitted)
            for layer, woke in woke_at.items():
                told[layer].append(woke)
            # One number per round, so the run can be checked for being one
            # run. See describe_drift.
            windows.append(max(woke_at.values()))
            # The layer is delivered when the *last* owner has been told, which
            # is the same "all ranks" boundary cq_last uses; mixing a first and
            # a last across the two columns would make idle look shorter than
            # any rank experienced.
            single.append([(layer, began[layer], ended[layer], first[layer],
                            last[layer], wrote.get(layer, last[layer]),
                            woke_at[layer])
                           for layer in sorted(last) if layer in woke_at])
            if failures:
                raise AssertionError(
                    f"round {round_id} marker={marker}: an owner's consuming kernel "
                    f"did not see the layer it was told was ready\n  " +
                    "\n  ".join(failures[:16]))
            if args.progress and (round_id + 1) % args.progress == 0:
                print(f"[visibility] round={round_id+1}/{args.rounds} "
                      f"elapsed={time.perf_counter()-started:.1f}s "
                      f"notify_to_launch={describe_gaps(gaps)}\n"
                      f"             layer_ready {describe_layer_ready(told, cq_last)}",
                      flush=True)

        stats = dict(group.gdr_statistics())
        assert stats["h2d.gdr.calls"] > 0 and stats["h2d.init_fallback.calls"] == 0
        flush_reads = stats.get("verbs.flush_reads", 0)
        if args.expect_flush:
            assert flush_reads > 0, (
                "no verbs.flush_reads: this build posted no NIC-side read-back, so "
                "the run says nothing about whether XDR needs one")
        if args.control_no_flush:
            # Without this assertion the control is a placebo: a typo in the
            # env var name would leave the read-back on, the run would pass,
            # and the pass would be read as "the read-back is unnecessary".
            assert flush_reads == 0, (
                f"--control-no-flush but verbs.flush_reads={flush_reads}: the "
                "read-back was still posted, so this run is not a control. Check "
                "FLEXKV_GDR_SKIP_FLUSH is set in the *worker* process")
        payload = args.rounds * args.layers * args.tp * args.kv_dim * len(TOUCHED) * args.fragment
        verdict = ("CONTROL RUN, not a gate: the read-back was disabled and no "
                   "mismatch appeared, which bounds the risk but does not show the "
                   "read-back is safe to remove -- memory ordering is not "
                   "established by failing to observe a violation"
                   if args.control_no_flush else
                   "every layer read by an owner-side kernel on notification, "
                   "byte-exact")
        print(f"[visibility] {args.rounds} rounds x {args.layers} layers x tp{args.tp} "
              f"({args.tp} owner processes): {verdict}; {payload/2**30:.2f}GiB; "
              f"notify_to_launch={describe_gaps(gaps)}; "
              f"verbs.flush_reads={flush_reads}; {stats}", flush=True)
        print(f"[visibility] consumer_wait={describe_gaps(waits)}; "
              "select/read wall time including scheduling and syscall overhead; "
              "synthetic consumer, not SGLang compute stall", flush=True)
        # A run where some layer quietly went via a CUDA event would produce a
        # perfectly plausible table measuring the driver instead of the fence.
        expected = args.rounds * args.layers
        assert args.notify != "polling" or stream_free == expected, (
            f"{stream_free}/{expected} layers were announced without touching a "
            "CUDA event: this transfer was not pure GDR, so the notification "
            "timings below include driver latency that production would not pay")
        # Before the tables, because it says whether they mean anything.
        drift = describe_drift(windows)
        if drift:
            print(f"[visibility] {drift}", flush=True)
        print(f"[visibility] layer_ready {describe_layer_ready(told, cq_last)}", flush=True)
        print(f"[visibility] "
              f"{describe_timeline(*columns, joined, entry_spread)}", flush=True)
        print(f"[visibility] {describe_round(single)}", flush=True)
        print(f"[visibility] slowest rank per layer: "
              f"{describe_laggards(laggards, args.tp)}", flush=True)
        print(f"[visibility] "
              f"{describe_ranks(rank_totals, rank_steps, rank_fences)}",
              flush=True)
    finally:
        group.wait_all_streams()
        group.configure_gdr(dict(config, enabled=False))
        close_leases(leases)
        for fd in fds:
            os.close(fd)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--qps", type=int, default=4)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--kv-dim", type=int, default=1, choices=[1, 2])
    p.add_argument("--fragment", type=int, default=64 << 10,
                   help="bytes per block per layer per KV component")
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--notify", default="polling", choices=["hostfunc", "polling"])
    p.add_argument("--background", action="store_true",
                   help="run owner-side compute on a side stream, as SGLang would")
    p.add_argument("--expect-flush", action="store_true",
                   help="require verbs.flush_reads > 0; use on the XDR build")
    p.add_argument("--control-no-flush", action="store_true",
                   help="negative control: require verbs.flush_reads == 0, as "
                        "FLEXKV_GDR_SKIP_FLUSH=1 should produce. Asks whether the "
                        "read-back is load-bearing; never a passing gate")
    p.add_argument("--progress", type=int, default=25, help="0 to silence")
    p.add_argument("--nics", default="")
    args = p.parse_args()
    try:
        validate_notify_mode(args.notify, _IS_XMLIR)
    except ValueError as exc:
        p.error(str(exc))
    assert hasattr(os, "eventfd"), "this test needs Linux eventfd (Python 3.10+)"
    assert not (args.expect_flush and args.control_no_flush), (
        "--expect-flush and --control-no-flush ask opposite questions")
    self_check()

    config = parse_gdr_config(dict(
        enabled=True, qp_count=args.qps,
        nic_names=args.nics.split(",") if args.nics else [],
        h2d=dict(path="gdr", on_init_failure="error"),
        d2h=dict(path="gdr", on_init_failure="error"),
        diagnostics=dict(summary=True, timing=False)))
    context = mp.get_context("spawn")
    conns, processes = [], []
    for rank in range(args.tp):
        parent, child = context.Pipe()
        process = context.Process(target=owner_process,
                                  args=(child, args, config, rank))
        process.start()
        child.close()
        conns.append(parent)
        processes.append(process)
    try:
        worker_process(conns, args, config)
        gc.collect()
        close_all_cuda_ipc_handles()
        for conn in conns:
            conn.send(("stop",))
        for conn in conns:
            assert receive(conn) == "owner MRs released"
        for rank, process in enumerate(processes):
            process.join(60)
            assert process.exitcode == 0, f"owner {rank} exited {process.exitcode}"
    except KeyboardInterrupt:
        # The teardown below is not instant -- MRs to release, eight processes
        # to reap -- and silence while it runs reads as a hang, which invites
        # the second Ctrl-C that lands in the middle of it.
        print("\n[visibility] interrupted; releasing owner registrations",
              flush=True)
        raise SystemExit(130) from None
    finally:
        # Owners ignore SIGINT, so this is where they end, and it has to hold up
        # under a second one. Closing the pipes first is what makes that true:
        # an owner blocked on recv sees EOF and leaves on its own, so an
        # interrupt part-way through the loop below still finds them exiting
        # rather than stranded. Signal every process before joining any, or the
        # timeouts add up across eight of them.
        for conn in conns:
            conn.close()
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(10)
            if process.is_alive():
                # SIGTERM is a request, and a process wedged in the device
                # driver can decline it. An unbounded join there is
                # indistinguishable from the hang this is escaping.
                process.kill()
                process.join(5)


if __name__ == "__main__":
    main()
