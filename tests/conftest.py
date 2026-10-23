"""Pure Python model fixtures; no federate or simulation runtime is imported."""
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in ("federates", "webapp"):
    sys.path.insert(0, str(ROOT / directory))

from cpslib.network import Network
from cpslib.placement import optimal_placement
from cpslib.pmu import PmuLayout
from run_experiment import build_configs, parse_args


@pytest.fixture(scope="session")
def repo_root():
    return ROOT


@pytest.fixture(scope="session")
def network_for():
    @lru_cache(maxsize=None)
    def load(case):
        return Network(str(ROOT / "cases" / f"ieee{case}" / "topology.json"))

    return load


@pytest.fixture(scope="session")
def solution_for(network_for):
    @lru_cache(maxsize=None)
    def solve(case):
        voltage, _, converged = network_for(case).solve_pf()
        assert converged, f"IEEE {case} power flow did not converge"
        return voltage

    return solve


@pytest.fixture(scope="session")
def layout_for(network_for):
    @lru_cache(maxsize=None)
    def layout(case, redundancy=1):
        net = network_for(case)
        buses, _ = optimal_placement(net, redundancy)
        assert len(buses) == len(set(buses))
        return PmuLayout(net, buses)

    return layout


@pytest.fixture
def ns3_noise():
    def sample(z, rng, mag=0.001, ang=0.001):
        # NS-3 Sample(): independent normal draws for magnitude and phase.
        return np.abs(z) * (1 + mag * rng.standard_normal(z.size)) * np.exp(
            1j * (np.angle(z) + ang * rng.standard_normal(z.size))
        )

    return sample


@pytest.fixture(scope="session")
def experiment_configs(tmp_path_factory):
    """Exercise the actual CLI -> NS-3/CC config path without starting federates."""
    import json

    @lru_cache(maxsize=None)
    def build(case=14, options=()):
        args = parse_args(["--case", str(case), *options])
        run_dir = tmp_path_factory.mktemp(f"config-{case}")
        cl = {"head": "localhost", "placement": {}, "nodes": [], "mpi": {}}
        meta = build_configs(args, str(run_dir), cl, 23500)
        ns3 = json.loads((run_dir / "ns3_config.json").read_text())
        run = json.loads((run_dir / "run_config.json").read_text())
        return ns3, run, meta, run_dir

    return build


@pytest.fixture(scope="session")
def run_request_model():
    """The real webapp RunRequest model, loaded from app.py's AST.

    Importing app.py needs FastAPI and starts job threads, so only the class is
    compiled, with the names app.py imports for it. One loader for all tests: a
    new name used by RunRequest only has to be added here.
    """
    import ast
    from typing import Optional

    from pydantic import BaseModel, Field, model_validator

    import scenario
    from cpslib import grids

    path = ROOT / "webapp" / "app.py"
    tree = ast.parse(path.read_text())
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RunRequest")
    namespace = {"BaseModel": BaseModel, "Field": Field, "model_validator": model_validator,
                 "Optional": Optional, "scen": scenario, "grids": grids}
    exec(compile(ast.Module(body=[model], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["RunRequest"]
