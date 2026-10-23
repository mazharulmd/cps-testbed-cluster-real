# Generator reactive limits

`Network.solve_pf(..., qlim=True)` now enforces aggregate reactive limits at
PV buses, summing Qmin/Qmax over **in-service** generators at each bus. Reference
buses remain exempt. After a converged Newton solve, every violating PV bus is
converted to PQ at its violated limit; the next solve starts from those voltages.
All violators switch together (MATPOWER `ENFORCE_Q_LIMS=1`), reducing the number
of solves when several machines violate independent limits. A switched bus stays
PQ for the rest of that call; the next call starts with the original bus types.
Neither the topology nor generator records are modified.

The return tuple remains `(V, it, ok)`. `it` counts Newton updates across rounds.
`max_it` still bounds each Newton solve; the new `max_qlim_it=30` bounds the total
number of power-flow rounds. Exhausting either bound returns `ok=False`, including
when Newton converged but there was no remaining round to enforce a newly found
limit. `last_qlim` resets on each call and maps switched bus IDs to `"max"` or
`"min"`. On failure it describes attempted switches; only a successful solve
certifies the final limits. Reactive violation tolerance is
`max(tol * baseMVA, 1e-7)` MVAr.

Experiments default to enforcement. Use `--no-qlim` in `run_experiment.py`, or
`qlim: false` in a scenario/API request, to disable it. Old scenario files and
`run_config.json` files without this field default to `true`. The option follows
base-case/attack planning, the built-in fallback, and GridPACK. For example:

```sh
PYTHONDONTWRITEBYTECODE=1 ~/.boss-venv/bin/python -B federates/run_experiment.py \
  --case 118 --solver builtin --event avr:49:1.12:4 --duration 10
# Add --no-qlim for the historical unconstrained operating point.
```

`grid_steps.csv` appends `q_limited`, a sorted, space-separated list such as
`19:min 49:max 103:max`. Existing columns retain their names and positions.
For the built-in solver these are its PV-to-PQ decisions. The GridPACK voltage
protocol has no switch flags, so its entries are **inferred contacts**: recompute
Qg from the returned voltages and actual step loads, and compare to aggregate
limits within 1e-3 MVAr. Contacts exclude reference buses and offline generators.
Uploaded networks whose parameters are rounded by the existing RAW writer may
have less accurate inferred contacts.

Dynamic mode applies this option to the **initial power flow** only. It does not
add machine over/under-excitation limiters to the dynamic models. Its `q_limited`
column is empty because there is no per-step PV/PQ switching state to report.

## Installed GridPACK evidence

The installed GridPACK implements reactive enforcement, but reads `qlim` as an
**integer**. The generated XML therefore uses `<qlim>1</qlim>` / `<qlim>0</qlim>`.
Merely replacing `false` with `true` would leave enforcement disabled in this
build. Both persistent-solver XML templates, shipped case XML files, and new case
generation use the integer form. No files under `/opt/cps` or C++ servers changed.

Header evidence (absolute paths on the review environment):

- `/opt/cps/gridpack/include/gridpack/applications/modules/powerflow/pf_app_module.hpp:874-875`:
  `// qlim enforce flag` followed by `int p_qlim;`.
- The same file at lines 261-273 declares `checkQlimViolations()` and
  `clearQlimViolations()`; its reset comment says “reset system to its original state”.
- `/opt/cps/gridpack/include/gridpack/applications/modules/powerflow/pf_factory_module.hpp:114-126`
  declares the network checks and reset, with `@return true if no violations found`.
- `/opt/cps/gridpack/include/gridpack/applications/components/pf_matrix/pf_components.hpp:185-190`
  documents `isPV()` as “V held fixed in powerflow equations”; lines 245-254
  declare `chkQlim()` and `clearQlim()`. Lines 459-475 hold generator status,
  Qmin/Qmax vectors and saved PV state.

Only headers and shared libraries are installed, not the corresponding `.cpp`
sources. To verify implementation rather than infer it from declarations, the
installed libraries were inspected with `nm -D -C` and `objdump -d -C`:

- `libgridpack_powerflow_module.so`, `PFAppModule::readNetwork`, address `0xf9446`,
  loads the string `qlim`; `0xf945d` calls `Configuration::get(string, int)` with
  default zero; `0xf9462` stores it in `p_qlim`.
- `libgridpack_configuration.so`, `Configuration::get(string, int)`, `0x172f1`,
  calls `std::istream::operator>>(int&)`. It uses integer parsing, not boolean parsing.
- `PFAppModule::solve`, `0xf6bf4-0xf6bf9`, tests that flag;
  `0xf7203` calls `PFFactoryModule::checkQlimViolations()`. A violation sets the
  repeat flag at `0xf721d`, and `0xf6e02-0xf6e05` returns to the solve loop.
- `libgridpack_pfmatrix_components.so`, `PFBus::chkQlim`, `0x10bb8-0x10c0d`,
  sums generator quantities for online status `1`. It checks upper/lower Q bounds
  at `0x10cb3` and `0x10cd8`, and clears the PV flag at `0x10d37` on a violation,
  fixing the net bus injection at the bound. It only processes PV buses
  (`0x10b55-0x10b70`).

These are observations of the installed binary, not source line claims. A real
MPI check was attempted, but the sandbox rejects sockets before server startup:
`opal_ifinit: socket() failed with errno=1`, then
`The PMIx server's listener thread failed to start`. Consequently, numerical
GridPACK enforcement and repeated-request reset still need an unsandboxed run.
The Python tests cover the XML, voltage-response diagnostics, federate step loop,
and built-in fallback. No C++ rebuild is required by this change.

## Reference data and validation

`tools/make_cases.py` originally generated `reference.csv` using the old,
unconstrained `Network.solve_pf()`, then compared it to MATPOWER's stored case
voltages. Those files therefore describe **no-limit** solutions. Its reference
function and `test_power_flow_matches_reference` now explicitly pass `qlim=False`.
No `reference.csv` was edited. A direct comparison with the original solver found
bit-for-bit identical complex voltages, iteration counts and success flags on all
six base cases and the IEEE 118 AVR case with `qlim=False`.

With enforcement, IEEE 118 `vset={49: 1.12}` produces Qg(49) =
210.000000000003 MVAr and |V49| = 1.018792924885 pu. Both F2 xfail markers have
been removed without changing their assertions or tolerances. The base-case
F2 test checks all six grids, excluding reference buses.

`tests/test_qlim.py` independently builds pandapower networks from the actual
`topology.json` records using `from_ppc`. It includes shunts, transformer taps,
phase shifts, loads, and generator limits/status, and checks both unconstrained
and `pandapower.runpp(..., enforce_q_lims=True)` results. Maximum magnitude errors:

| Case | No limits (pu) | Limits enforced (pu) |
| --- | ---: | ---: |
| IEEE 14 base | 4.441e-16 | 4.441e-16 |
| IEEE 118 base | 9.992e-16 | 6.661e-16 |
| IEEE 118 AVR 49 at 1.12 | 1.443e-15 | 8.882e-16 |

Pandapower tests use `pytest.importorskip` if that optional package is absent.
The supplied venv already includes it; no installation is needed.

To reproduce:

```sh
PYTHONDONTWRITEBYTECODE=1 ~/.boss-venv/bin/python -B -m pytest -q
PYTHONDONTWRITEBYTECODE=1 ~/.boss-venv/bin/python -B -m pytest -q -s tests/test_qlim.py
```

Regression evidence was obtained using an isolated `git archive HEAD` under
`/tmp/wp3-original`, with only the new tests and unmarked F2 tests copied in.
The production worktree was never replaced. Original-code output included:

```text
E AssertionError: IEEE 39 bus 37: -1.369447 MVAr outside [0.0, 250.0]
E assert np.float64(746.7209872459842) <= (210 + 0.001)
E Obtained: 746.7209872459842
E Expected: 210 ± 0.001
E ValueError: qlim: Extra inputs are not permitted
E TypeError: GridPackMPISolver.__init__() got an unexpected keyword argument 'qlim'
E - q_limited
E + gridpack_solve_ms
24 failed, 2 deselected in 2.68s
```

This covers both F2 tests and every parametrization in
`test_qlim.py`: AVR saturation/reset, iteration caps, aggregate online limits
with changed loads, lower limits/slack exclusion, pandapower comparisons,
CLI/scenario configuration, GridPACK XML/diagnostics, per-step AVR CSV logs, missing-option defaults, MPI failure fallback, and Web API
argument forwarding.

Final full-suite output (4.68 s wall time including interpreter startup):

```text
65 passed, 3 xfailed in 3.70s
```

The three remaining strict xfails cover unrelated audit findings. The full
MPI/HELICS/NS-3 AVR scenario still needs the reviewer run outside this sandbox.
