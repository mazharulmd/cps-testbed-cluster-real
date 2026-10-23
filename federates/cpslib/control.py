"""Voltage-control safety, generator selection and protocol bookkeeping; no HELICS."""
from collections import deque


def nearest_generators(net, gen_buses):
    """Hop distance from every bus to the closest in-service generator bus."""
    adj = {b: set() for b in net.bus_ids}
    for f, t in zip(net.f, net.t):
        adj[net.bus_ids[f]].add(net.bus_ids[t]); adj[net.bus_ids[t]].add(net.bus_ids[f])
    best = {}
    for b in net.bus_ids:
        seen, q = {b}, deque([(b, 0)])
        while q:
            x, d = q.popleft()
            if x in gen_buses:
                best[b] = x
                break
            for y in adj[x]:
                if y not in seen:
                    seen.add(y); q.append((y, d + 1))
    return best


def control_decision(result, *, enabled=True, cooldown=False):
    """Return a hold reason, or an empty string when the final estimate permits control.

    Use the estimate *after* BDD removal. An unobservable estimate can have alarm=False
    because the chi-square test does not apply to its pseudo-measurement solution.
    """
    if not enabled:
        return "disabled"
    if not result.get("observable", False):
        return "unobservable"
    if result.get("pseudo_measurements", 0):
        return "pseudo_measurements"
    if result.get("alarm", True):
        return "bad_data"
    if cooldown:
        return "cooldown"
    return ""


def default_ack_timeout(max_link_delay_ms, grid_step, rate):
    """Twice the expected maximum application round trip (seconds).

    Include both generator link delays, one grid step, four NS-3/HELICS frame
    handoffs, and serialization of two 1500-byte packets at the default 10 Mbps.
    Deliberate attack delays and arbitrary queueing are outside this budget.
    """
    return 2 * (2 * max_link_delay_ms / 1000 + grid_step + 4 / rate + 0.0024)


class SetpointTelemetry:
    """Last delivered generator state. Sending commands never changes this state."""
    def __init__(self, initial):
        self.vset = dict(initial)
        self.q_mvar = {}
        self.q_pu_by_machine = {}
        self.times = {}

    def update(self, packet):
        bus, t = int(packet["gen_bus"]), float(packet["t"])
        if bus not in self.vset or t < self.times.get(bus, -float("inf")):
            return False
        self.vset[bus] = float(packet["vset"])
        self.times[bus] = t
        if packet.get("q_mvar") is not None:
            self.q_mvar[bus] = float(packet["q_mvar"])
        if "q_pu_by_machine" in packet:
            self.q_pu_by_machine[bus] = dict(packet["q_pu_by_machine"])
        return True


class AppliedCommands:
    """Grid-side idempotence: retransmissions get an ack without a second application."""
    def __init__(self):
        self.applied = {}

    def receive(self, command, now):
        ident = command["id"]
        if ident in self.applied:
            return False, dict(self.applied[ident])
        ack = {"id": ident, "gen_bus": command["gen_bus"], "vset": command["vset"],
               "applied_t": now}
        self.applied[ident] = ack
        return True, dict(ack)


class CommandTracker:
    """CC command lifecycle, in simulation time; one retry with the same id.

    Keep completed ids to distinguish duplicate/late acks from unknown (forged)
    ids. Late acks reclassify an unacknowledged command as acknowledged. Counters
    describe logical commands, except unexpected_acks which counts packets.
    """
    def __init__(self, timeout):
        import math
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("ack timeout must be finite and positive")
        self.timeout = timeout
        self.records = {}
        self.events = []
        self.counts = dict(commands_acked=0, commands_retried=0,
                           commands_unacked=0, unexpected_acks=0)

    def _event(self, event, command, now, attempt="", reason=""):
        self.events.append(dict(t=now, event=event, id=command["id"],
                                gen_bus=command["gen_bus"], attempt=attempt, reason=reason))

    def issue(self, command, now):
        if command["id"] in self.records:
            raise ValueError("command id already issued")
        self.records[command["id"]] = dict(command=dict(command), deadline=now + self.timeout,
                                          attempt=1, status="pending")
        self._event("issued", command, now, 1)

    def pending_for(self, bus):
        return any(r["status"] == "pending" and r["command"]["gen_bus"] == bus
                   for r in self.records.values())

    def next_deadline(self, end):
        return min([end] + [r["deadline"] for r in self.records.values() if r["status"] == "pending"])

    def acknowledge(self, ack, now):
        record = self.records.get(ack["id"])
        if record is None or record["command"]["gen_bus"] != ack["gen_bus"]:
            self.counts["unexpected_acks"] += 1
            self._event("unexpected_ack", ack, now)
            return "unexpected_ack"
        status = record["status"]
        event = "duplicate_ack" if status == "acked" else "late_ack" if status == "unacked" else "acked"
        if status != "acked":
            self.counts["commands_acked"] += 1
            if status == "unacked":
                self.counts["commands_unacked"] -= 1
            record["status"] = "acked"
        self._event(event, ack, now, record["attempt"])
        return event

    def _unacked(self, record, now, reason):
        record["status"] = "unacked"
        self.counts["commands_unacked"] += 1
        self._event("unacknowledged", record["command"], now, record["attempt"], reason)

    def expire(self, now):
        retries = []
        for record in self.records.values():
            if record["status"] != "pending" or now < record["deadline"] - 1e-9:
                continue
            if record["attempt"] == 1:
                record.update(attempt=2, deadline=now + self.timeout)
                retries.append(dict(record["command"], attempt=2))
                self.counts["commands_retried"] += 1
                self._event("retried", record["command"], now, 2)
            else:
                self._unacked(record, now, "timeout")
        return retries

    def finish(self, now):
        for record in self.records.values():
            if record["status"] == "pending":
                self._unacked(record, now, "run_end")


def command_summary(events):
    """Derive logical outcomes from cc_commands.csv, including late/duplicate acks."""
    terminal, retried, unexpected = {}, set(), 0
    for event in events:
        ident, kind = event["id"], event["event"]
        if kind in ("acked", "late_ack", "duplicate_ack"):
            terminal[ident] = "acked"
        elif kind == "unacknowledged":
            terminal[ident] = "unacked"
        elif kind == "retried":
            retried.add(ident)
        elif kind == "unexpected_ack":
            unexpected += 1
    return dict(commands_acked=sum(v == "acked" for v in terminal.values()),
                commands_retried=len(retried), commands_unacked=sum(v == "unacked" for v in terminal.values()),
                unexpected_acks=unexpected)
