# Control-center safety and generator feedback

The controller uses the **final** estimate after any bad-data removal. New
commands require observability, zero pseudo-measurements, and no final
chi-square alarm. `alarm` still means an observable estimate failed its test;
unobservable estimates still have `alarm=false`. The appended estimator field
`pseudo_measurements` records the number of weak priors used. A successful BDD
cleanup can permit control; a cleanup that destroys observability cannot.

`cc_log.csv` retains every existing column and appends `hold_reason` and
`pseudo_measurements`. The reason is empty when the set produces a new command.
Reasons include `unobservable`, `pseudo_measurements`, `bad_data`, `disabled`,
`cooldown`, `pending_ack`, `setpoint_limit`, `no_generator`, `deadband`, and
`no_violation`. When several generators are blocked, the last evaluated reason
is recorded. `summary.json.control_center.hold_reason_counts` counts sets by
reason, including sustained bad-data holds during replay. Existing detection
rates still use `first_alarm`; their meanings have not changed.

## Transport and applied state

The grid publishes `grid/generators` every grid step, containing the sample time,
actual applied setpoints, available Q, and command acknowledgements. Only NS-3
subscribes to this publication. Each generator node has a reverse UDP socket
connected to the CC's port 5001 over its existing point-to-point link. NS-3
schedules telemetry every `telemetry_s`, starting when generator state first
arrives, and immediately sends each grid acknowledgement into that link. Only
after packet reception at the CC node does NS-3 publish `ns3/generators` for the
CC federate. There is no direct grid-to-CC shortcut.

Telemetry retains the original grid sample time; its age includes the grid
sampling interval, HELICS handoffs and generator-link delay. The CC starts with
case setpoints until feedback arrives. Delivered telemetry replaces the known
setpoint, including changes from AVR faults and forged commands. Sending a
command does not update that known applied value. Acknowledgements also report
the applied value and time; they update known state only when newer than the
latest accepted state. Older telemetry cannot overwrite newer feedback.

QSS telemetry reports aggregate generator reactive output per bus as `q_mvar`,
computed from solved network injection plus local reactive demand. Dynamic mode
passes through dsf_server's available online-machine watch Q as
`q_pu_by_machine` (machine id to raw pu value). Watch-value bases may depend on
the machine model, so these values are neither summed nor relabelled MVAr.
`cc_telemetry.csv` logs delivered telemetry, its sample/delivery times, Q, and
whether it was accepted. State/Q are diagnostic; voltage control uses setpoints.

The existing generator links have propagation delay and serialization. The
existing configured random packet-error model remains on PMU receive links.
`cmd-drop` and `cmd-delay` intercept **commands only**, including retries, at
network ingress during the attack window. They do not intercept reverse
telemetry/acks. `cmd-inject` and coordinated attacks still send through the
normal CC command socket, bypassing CC pending bookkeeping; their applied ids
produce `unexpected_ack` at the CC and their setpoints appear in feedback.

## Configuration and acknowledgement lifecycle

CLI options, scenario `control:` fields, and the web API accept:

- `telemetry_s` / `--telemetry-s`: positive finite seconds, default **2.0**.
- `ack_timeout_s` / `--ack-timeout-s`: positive finite seconds; omitted/null
  derives **twice the maximum expected application round trip**.

The derived timeout is `2 * (2*max_generator_delay_ms/1000 + grid_step +
4/rate + 0.0024)`: both generator propagation delays, one worst-case grid-step
wait, four NS-3/HELICS frame handoffs and serialization of two 1500-byte packets
at the configured default 10 Mbps. Attack-added delay and arbitrary congestion
are outside this budget; use an explicit override for those experiments.
The resolved timeout is persisted in run config, metadata, and summary. Legacy
handwritten CC configs without a timeout use 2 seconds; old NS-3 configs without
a telemetry interval use 2 seconds. Existing scenario files need no changes.

Every new CC command enters the pending set. The CC processes delivered acks
before checking deadlines and requests its next HELICS grant at the nearest
pending deadline even if PDC traffic stops. At the first deadline it retries
once, keeping the id, value, original issue time, and reason; `attempt: 2` makes
the publication distinct from the original for NS-3 duplicate-payload handling.
At the second deadline it logs `unacknowledged`. Retries continue the lifecycle
of a previously trusted command independently of new-estimate holds. A generator
with a pending command cannot receive a newly computed command. The grid applies
each id once and re-acknowledges delivered retries without reapplying them.

`cc_commands.csv` records `issued`, `retried`, `acked`, `unacknowledged`,
`unexpected_ack`, `duplicate_ack`, and `late_ack` events with id, generator,
simulation time, attempt, and reason. A late known ack changes the final outcome
from unacknowledged to acknowledged. Duplicate known acks do not increment
success counts; unknown ids (or mismatched generator buses) increment
`unexpected_acks`. Identical re-reported HELICS feedback payloads are ignored.
Commands still pending at simulation end are logged unacknowledged with reason
`run_end`; these may have less than a full timeout budget remaining. Timed-out
commands use reason `timeout`. These categories are distinguishable in the CSV.

`summary.json.commands` appends `commands_acked`, `commands_retried`,
`commands_unacked`, and `unexpected_acks`, derived from that lifecycle log. The
first three count unique CC commands (retries can also ultimately be acked or
unacked); unexpected acks count delivered packets. Old runs without the log
report zero new counters. `commands_applied.csv` appends `id`, blank for events.
`ns3_summary.json` appends `telemetry_sent`, `telemetry_delivered`, `acks_sent`,
and `acks_delivered`, counting individual UDP packets; original command and
attack counters retain their meanings. `frames.csv` is unchanged.

## Validation

Run the pure Python suite without writing bytecode:

```sh
PYTHONDONTWRITEBYTECODE=1 ~/.boss-venv/bin/python -B -m pytest -q
```

Tests use real IEEE 14 estimates and power flows plus an in-memory HELICS double
for Python loop checks. They cannot validate C++ compilation, UDP event timing,
or real HELICS grants. Reviewer checks should compile with ns-3.48, run AVR
faults with no attack, `cmd-drop`, and `cmd-inject`, and inspect generator
feedback, command lifecycle logs, and summary counters. Include a run long
enough for both command timeout intervals, and a replay run whose final
estimates alarm to verify visible `bad_data` holds throughout the attack.
