"""
Self-organizing field: an online graph that forms its own states from a stream.

The rule is ``observe -> resonate or spawn -> reweight -> compose motifs``. There
is no preset cluster count, no batch fitting, and no HDBSCAN: each observation is
either close enough to an existing prototype to reinforce it, or it becomes a new
one. That matters because a batch clusterer has to see the whole stream before it
can say anything, which is the wrong shape for something meant to run on a live
input and keep a persistent memory.

Two layers are kept deliberately separate:

  - what kinds of things exist: prototypes, formed from feature geometry alone
  - how those things organize in time: edges and motifs over the state sequence

Shuffling a stream preserves the first and destroys the second, which is what
makes the null test in agents/emergence.py meaningful.

Nothing here uses a model. Feature extraction is the caller's job; this operates
on vectors, so the same field works for audio events, image descriptors or
anything else measurable.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# How close an observation must be to a prototype to reinforce it rather than
# spawn a new one, in units of standard deviations after whitening. Vigilance in
# the ART sense: lower spawns more states, higher merges more aggressively.
DEFAULT_VIGILANCE = 1.5

# Weight of a new observation when updating a prototype centroid. Low enough that
# one outlier cannot drag a well-supported state, high enough to track drift.
DEFAULT_LEARNING_RATE = 0.12

# Edge weights decay per observation so that a transition seen once long ago does
# not count the same as one still recurring.
DEFAULT_EDGE_DECAY = 0.999


@dataclass
class Prototype:
    """One self-formed state: a centroid plus the evidence supporting it."""

    id: str
    centroid: np.ndarray
    support: int = 1
    created_at_index: int = 0
    last_seen_index: int = 0
    # Mean distance of assigned observations from the centroid, as a running
    # figure. A tight prototype is a more trustworthy object than a diffuse one.
    mean_radius: float = 0.0

    def stability(self, total_observations: int) -> float:
        """
        How settled this state is, in [0, 1].

        Combines how much evidence it has with how tight it is. A state seen once
        is not stable no matter how tight, and a state seen often but sprawling is
        not a clean object either.
        """
        if total_observations <= 0:
            return 0.0
        evidence = min(1.0, self.support / max(4.0, 0.02 * total_observations))
        tightness = 1.0 / (1.0 + self.mean_radius)
        return round(evidence * tightness, 4)


@dataclass
class Motif:
    """A recurring subsequence of states."""

    id: str
    sequence: Tuple[str, ...]
    support: int

    @property
    def level(self) -> int:
        return len(self.sequence)


@dataclass
class SelfOrganizingField:
    """
    An online field of prototypes with a temporal graph over them.

    Feed it observations one at a time with :meth:`observe`. It keeps running
    feature statistics so distances are scale-free without a separate fitting
    pass, which is what lets it start cold on a live stream.
    """

    vigilance: float = DEFAULT_VIGILANCE
    learning_rate: float = DEFAULT_LEARNING_RATE
    edge_decay: float = DEFAULT_EDGE_DECAY
    stream_id: str = "stream"

    prototypes: Dict[str, Prototype] = field(default_factory=dict)
    edges: Dict[Tuple[str, str], float] = field(default_factory=dict)
    edge_counts: Dict[Tuple[str, str], int] = field(default_factory=dict)
    sequence: List[str] = field(default_factory=list)

    _n: int = 0
    _mean: Optional[np.ndarray] = None
    _m2: Optional[np.ndarray] = None
    _next_id: int = 0

    # ---- whitening -------------------------------------------------------

    def _update_feature_stats(self, x: np.ndarray) -> None:
        """Welford update, so distances are comparable across feature scales."""
        self._n += 1
        if self._mean is None:
            self._mean = np.zeros_like(x, dtype=float)
            self._m2 = np.zeros_like(x, dtype=float)
        delta = x - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (x - self._mean)

    def _whiten(self, x: np.ndarray) -> np.ndarray:
        if self._mean is None or self._n < 2:
            return x.astype(float)
        variance = self._m2 / max(1, self._n - 1)
        scale = np.sqrt(np.maximum(variance, 1e-8))
        return (x - self._mean) / scale

    # ---- the rule --------------------------------------------------------

    def observe(self, vector: Sequence[float]) -> str:
        """
        Take one observation and return the id of the state it belongs to.

        Resonates with the nearest prototype when within vigilance, otherwise
        spawns a new one. Either way the transition from the previous state is
        reinforced and every other edge decays a little.
        """
        x = np.asarray(vector, dtype=float)
        self._update_feature_stats(x)
        wx = self._whiten(x)

        state_id, distance = self._nearest(wx)
        if state_id is None or distance > self.vigilance:
            state_id = self._spawn(wx)
        else:
            self._resonate(state_id, wx, distance)

        if self.sequence:
            self._reinforce(self.sequence[-1], state_id)
        self.sequence.append(state_id)
        return state_id

    def observe_many(self, vectors: Iterable[Sequence[float]]) -> List[str]:
        return [self.observe(v) for v in vectors]

    def _nearest(self, wx: np.ndarray) -> Tuple[Optional[str], float]:
        best_id, best_distance = None, math.inf
        for pid, proto in self.prototypes.items():
            distance = float(np.linalg.norm(wx - proto.centroid))
            if distance < best_distance:
                best_id, best_distance = pid, distance
        return best_id, best_distance

    def _spawn(self, wx: np.ndarray) -> str:
        pid = f"s{self._next_id}"
        self._next_id += 1
        index = len(self.sequence)
        self.prototypes[pid] = Prototype(
            id=pid, centroid=wx.copy(), support=1,
            created_at_index=index, last_seen_index=index,
        )
        return pid

    def _resonate(self, pid: str, wx: np.ndarray, distance: float) -> None:
        proto = self.prototypes[pid]
        proto.centroid = (1 - self.learning_rate) * proto.centroid + self.learning_rate * wx
        proto.support += 1
        proto.last_seen_index = len(self.sequence)
        # Running mean of assignment distance, so mean_radius reflects all the
        # evidence rather than only the most recent point.
        proto.mean_radius += (distance - proto.mean_radius) / proto.support

    def _reinforce(self, source: str, target: str) -> None:
        for key in list(self.edges):
            self.edges[key] *= self.edge_decay
        edge = (source, target)
        self.edges[edge] = self.edges.get(edge, 0.0) + 1.0
        self.edge_counts[edge] = self.edge_counts.get(edge, 0) + 1

    # ---- structure -------------------------------------------------------

    def merge_close_prototypes(self, threshold: Optional[float] = None) -> List[Tuple[str, str]]:
        """
        Merge prototypes that have drifted together, returning the pairs merged.

        Online spawning can create two states for what turns out to be one thing,
        when early observations arrive before the centroid settles. Merging is a
        separate explicit step so that a caller can decide when the stream is
        stable enough to consolidate.
        """
        threshold = self.vigilance * 0.5 if threshold is None else threshold
        merged: List[Tuple[str, str]] = []
        ids = sorted(self.prototypes, key=lambda p: -self.prototypes[p].support)

        for i, keep_id in enumerate(ids):
            if keep_id not in self.prototypes:
                continue
            for drop_id in ids[i + 1:]:
                if drop_id not in self.prototypes or drop_id == keep_id:
                    continue
                keep, drop = self.prototypes[keep_id], self.prototypes[drop_id]
                if float(np.linalg.norm(keep.centroid - drop.centroid)) > threshold:
                    continue
                total = keep.support + drop.support
                keep.centroid = (keep.centroid * keep.support + drop.centroid * drop.support) / total
                keep.support = total
                keep.last_seen_index = max(keep.last_seen_index, drop.last_seen_index)
                del self.prototypes[drop_id]
                self._rewrite_state(drop_id, keep_id)
                merged.append((drop_id, keep_id))
        return merged

    def _rewrite_state(self, old: str, new: str) -> None:
        self.sequence = [new if s == old else s for s in self.sequence]
        for mapping in (self.edges, self.edge_counts):
            for (a, b), value in list(mapping.items()):
                if a != old and b != old:
                    continue
                del mapping[(a, b)]
                key = (new if a == old else a, new if b == old else b)
                mapping[key] = mapping.get(key, type(value)()) + value

    def motifs(self, level: int = 3, min_support: int = 2) -> List[Motif]:
        """Recurring subsequences of the given length, most supported first."""
        if level < 2 or len(self.sequence) < level:
            return []
        counts: Dict[Tuple[str, ...], int] = {}
        for i in range(len(self.sequence) - level + 1):
            gram = tuple(self.sequence[i:i + level])
            counts[gram] = counts.get(gram, 0) + 1
        found = [(gram, n) for gram, n in counts.items() if n >= min_support]
        found.sort(key=lambda item: (-item[1], item[0]))
        return [Motif(id=f"m{i}", sequence=gram, support=n) for i, (gram, n) in enumerate(found)]

    def edge_list(self) -> List[Dict[str, Any]]:
        """Edges with both the decayed weight and the raw count behind them."""
        total = sum(self.edge_counts.values()) or 1
        out = []
        for (source, target), count in sorted(self.edge_counts.items(), key=lambda kv: -kv[1]):
            out.append({
                "from": source,
                "to": target,
                "count": count,
                "weight": round(self.edges.get((source, target), 0.0), 4),
                "share": round(count / total, 4),
            })
        return out

    def summary(self) -> Dict[str, Any]:
        total = len(self.sequence)
        return {
            "stream_id": self.stream_id,
            "observations": total,
            "states": len(self.prototypes),
            "edges": len(self.edge_counts),
            "state_support": {pid: p.support for pid, p in
                              sorted(self.prototypes.items(), key=lambda kv: -kv[1].support)},
            "vigilance": self.vigilance,
        }

    # ---- canonical form --------------------------------------------------

    def to_sexp(self, motif_level: int = 3, motif_stats: Optional[Dict[str, Any]] = None,
                edge_stats: Optional[Dict[Tuple[str, str], float]] = None) -> str:
        """
        Emit the field as a canonical S-expression.

        This is the interchange form: compact, unambiguous, and the same shape
        whether it came from audio, images or telemetry. Statistics computed
        elsewhere (null z-scores, motif p-values) can be passed in and are only
        included when they were actually measured.
        """
        total = len(self.sequence)
        lines = [f"(stream {self.stream_id}"]

        for pid, proto in sorted(self.prototypes.items(), key=lambda kv: -kv[1].support):
            lines.append(
                f"  (state (id {pid}) (support {proto.support}) "
                f"(stability {proto.stability(total):.3f}) "
                f"(radius {proto.mean_radius:.3f}))"
            )

        for edge in self.edge_list():
            parts = [
                f"(from {edge['from']})", f"(to {edge['to']})", "(kind temporal)",
                f"(count {edge['count']})", f"(share {edge['share']:.3f})",
            ]
            if edge_stats and (edge["from"], edge["to"]) in edge_stats:
                parts.append(f"(null-z {edge_stats[(edge['from'], edge['to'])]:.2f})")
            lines.append("  (edge " + " ".join(parts) + ")")

        for motif in self.motifs(level=motif_level):
            parts = [
                f"(id {motif.id})",
                "(sequence " + " ".join(motif.sequence) + ")",
                f"(support {motif.support})",
                f"(level {motif.level})",
            ]
            if motif_stats and motif.id in motif_stats:
                for key, value in motif_stats[motif.id].items():
                    parts.append(f"({key} {value})")
            lines.append("  (motif " + " ".join(parts) + ")")

        lines.append(f"  (observed {total}) (generated-at {int(time.time())}))")
        return "\n".join(lines)
