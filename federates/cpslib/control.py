"""Generator selection for voltage control (independent of HELICS)."""
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

