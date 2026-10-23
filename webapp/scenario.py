"""Scenario files: a whole experiment described in one YAML or JSON file that users upload.

See scenario_template.yaml for every field. A file holds one scenario, or several under
`scenarios:` (each one is queued as its own experiment; fields given at the top level are
defaults for all of them).
"""
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Pmu(Strict):
    placement: Literal["minimum", "redundant"] = "minimum"
    rate: float = Field(30, ge=1, le=120)


class NetworkCfg(Strict):
    latency_ms: float = Field(10, ge=0.1, le=2000)
    latency_spread: float = Field(0.5, ge=0, le=0.95)
    jitter_ms: float = Field(2, ge=0, le=500)
    loss: float = Field(0, ge=0, le=0.9)
    pdc_wait_ms: float = Field(20, ge=1, le=2000)


DYNAMIC_EVENTS = ("bus_fault", "line_trip", "gen_trip")


class Event(Strict):
    type: Literal["none", "avr_fault", "load_step", "bus_fault", "line_trip", "gen_trip"] = "none"
    bus: Optional[int] = None                         # generator, load or fault bus; "from" bus of a line
    to: Optional[int] = None                          # line_trip: "to" bus
    vset: float = Field(1.12, ge=0.8, le=1.3)         # avr_fault
    pct: float = Field(20, ge=-90, le=500)            # load_step
    duration: float = Field(0.1, gt=0, le=5)          # bus_fault: seconds until it is cleared
    t: float = Field(4, ge=0)

    @model_validator(mode="after")
    def _bus(self):
        if self.type != "none" and self.bus is None:
            raise ValueError(f"event '{self.type}' needs a bus")
        if self.type == "line_trip" and self.to is None:
            raise ValueError("event 'line_trip' needs bus and to (the two ends of the line)")
        return self


class Attack(Strict):
    type: Literal["none", "fdi-simple", "fdi-stealthy", "delay", "drop"] = "none"
    target: Optional[int] = None
    fake: Optional[float] = Field(None, ge=0.5, le=1.5)
    start: float = Field(4, ge=0)
    end: float = Field(9, ge=0)
    delay_ms: float = Field(100, ge=0, le=5000)

    @model_validator(mode="after")
    def _window(self):
        if self.type != "none" and self.end <= self.start:
            raise ValueError("attack end must be after attack start")
        return self


class Control(Strict):
    bdd: bool = True
    voltage_control: bool = True
    vmin: float = Field(0.94, ge=0.5, le=1.0)
    vmax: float = Field(1.08, ge=1.0, le=1.5)


class Scenario(Strict):
    name: str = Field("scenario", pattern=r"^[A-Za-z0-9_-]{1,40}$")
    grid: str = Field("ieee14", pattern=r"^[A-Za-z0-9_-]{1,48}$")
    duration: float = Field(10, ge=1, le=120)
    mode: Literal["qss", "dynamic"] = "qss"
    grid_step: Optional[float] = Field(None, ge=0.005, le=5)
    solver: Literal["gridpack", "builtin"] = "gridpack"
    mpi_np: Optional[int] = Field(None, ge=1, le=1024)     # default: the cluster setting
    seed: int = Field(1, ge=1, le=10 ** 6)
    pmu: Pmu = Pmu()
    network: NetworkCfg = NetworkCfg()
    event: Event = Event()
    attack: Attack = Attack()
    control: Control = Control()
    qlim: bool = True

    @model_validator(mode="after")
    def _mode(self):
        if self.mode == "qss" and self.event.type in DYNAMIC_EVENTS:
            raise ValueError(f"event '{self.event.type}' needs mode: dynamic")
        if self.mode == "dynamic" and self.event.type == "load_step":
            raise ValueError("event 'load_step' is available with mode: qss only")
        if self.mode == "dynamic" and self.solver != "gridpack":
            raise ValueError("mode: dynamic needs solver: gridpack")
        return self

    def run_args(self):
        """Fields of the experiment API's RunRequest."""
        ev = "none"
        if self.event.type == "avr_fault":
            ev = f"avr:{self.event.bus}:{self.event.vset}:{self.event.t}"
        elif self.event.type == "load_step":
            ev = f"load:{self.event.bus}:{self.event.pct}:{self.event.t}"
        elif self.event.type == "bus_fault":
            ev = f"fault:{self.event.bus}:{self.event.duration}:{self.event.t}"
        elif self.event.type == "line_trip":
            ev = f"line:{self.event.bus}:{self.event.to}:{self.event.t}"
        elif self.event.type == "gen_trip":
            ev = f"gen:{self.event.bus}:{self.event.t}"
        return {
            "case": self.grid, "name": self.name, "duration": self.duration, "mode": self.mode,
            "grid_step": self.grid_step,
            "rate": self.pmu.rate, "solver": self.solver, "mpi_np": self.mpi_np,
            "placement": 1 if self.pmu.placement == "minimum" else 2,
            "latency": self.network.latency_ms, "latency_spread": self.network.latency_spread,
            "jitter": self.network.jitter_ms, "loss": self.network.loss, "pdc_wait": self.network.pdc_wait_ms,
            "attack": self.attack.type, "target": self.attack.target, "fake": self.attack.fake,
            "attack_start": self.attack.start, "attack_end": self.attack.end, "attack_delay": self.attack.delay_ms,
            "bdd": "on" if self.control.bdd else "off", "control": "on" if self.control.voltage_control else "off",
            "vmin": self.control.vmin, "vmax": self.control.vmax, "event": ev, "seed": self.seed, "qlim": self.qlim,
        }


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def parse(text):
    """Scenario objects from the text of an uploaded file. Raises ValueError with a readable message."""
    try:
        doc = yaml.safe_load(text)          # YAML is a superset of JSON
    except yaml.YAMLError as e:
        raise ValueError(f"not valid YAML/JSON: {e}")
    if not isinstance(doc, dict):
        raise ValueError("the file must contain a scenario (a mapping of fields)")
    items: List[dict] = [doc]
    if "scenarios" in doc:
        common = {k: v for k, v in doc.items() if k != "scenarios"}
        if not isinstance(doc["scenarios"], list) or not doc["scenarios"]:
            raise ValueError("'scenarios' must be a non-empty list")
        if len(doc["scenarios"]) > 50:
            raise ValueError("at most 50 scenarios per file")
        items = [_merge(common, s or {}) for s in doc["scenarios"]]
    out = []
    for i, item in enumerate(items):
        try:
            out.append(Scenario(**item))
        except ValidationError as e:
            where = f"scenario {i + 1}: " if len(items) > 1 else ""
            msgs = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'scenario'}: {err['msg']}" for err in e.errors())
            raise ValueError(where + msgs)
        except TypeError as e:
            raise ValueError(str(e))
    return out
