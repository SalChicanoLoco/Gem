"""
Structure statistics over a state sequence, and the null they must beat.

The claim worth testing is not "clusters exist" but "the order of the states
carries information". Shuffling the sequence keeps every state and every state's
frequency and destroys only the temporal arrangement, so a statistic that stays
the same under shuffling was never measuring temporal structure.

Emergence at scale k is defined as the excess of measured structure over what the
shuffled null produces at that same scale:

    E_k = structure_k(real) - mean structure_k(shuffled)

Structure is oriented so that more is more organized: predictability and mutual
information as they are, entropies negated. E_k near zero means the level has not
earned promotion, however interesting the level below it looked.

p-values are empirical and bounded by the number of shuffles: with n shuffles the
smallest reportable value is 1/(n+1), and it is reported as such rather than
rounded to zero.
"""

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _bigrams(sequence: Sequence[str]) -> List[Tuple[str, str]]:
    return list(zip(sequence[:-1], sequence[1:]))


def transition_mutual_information(sequence: Sequence[str]) -> float:
    """
    I(X_t ; X_{t+1}) in bits.

    Zero when the next state is independent of the current one, which is what
    shuffling produces apart from finite-sample noise.
    """
    pairs = _bigrams(sequence)
    if not pairs:
        return 0.0
    n = len(pairs)
    joint = Counter(pairs)
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)

    total = 0.0
    for (a, b), count in joint.items():
        p_ab = count / n
        p_a = left[a] / n
        p_b = right[b] / n
        if p_ab > 0 and p_a > 0 and p_b > 0:
            total += p_ab * math.log2(p_ab / (p_a * p_b))
    return total


def next_state_predictability(sequence: Sequence[str]) -> float:
    """
    Fraction of transitions predicted by always guessing the most likely
    successor of the current state. A baseline of "guess the commonest state
    overall" is what shuffling reduces this to.
    """
    pairs = _bigrams(sequence)
    if not pairs:
        return 0.0
    successors: Dict[str, Counter] = {}
    for a, b in pairs:
        successors.setdefault(a, Counter())[b] += 1
    correct = sum(counter.most_common(1)[0][1] for counter in successors.values())
    return correct / len(pairs)


def conditional_entropy(sequence: Sequence[str]) -> float:
    """H(X_{t+1} | X_t) in bits. Lower means the past constrains the future more."""
    pairs = _bigrams(sequence)
    if not pairs:
        return 0.0
    n = len(pairs)
    successors: Dict[str, Counter] = {}
    for a, b in pairs:
        successors.setdefault(a, Counter())[b] += 1

    total = 0.0
    for a, counter in successors.items():
        p_a = sum(counter.values()) / n
        inner = 0.0
        for count in counter.values():
            p = count / sum(counter.values())
            inner -= p * math.log2(p)
        total += p_a * inner
    return total


def kgram_entropy(sequence: Sequence[str], k: int) -> float:
    """Shannon entropy of the k-gram distribution, in bits."""
    if k < 1 or len(sequence) < k:
        return 0.0
    grams = Counter(tuple(sequence[i:i + k]) for i in range(len(sequence) - k + 1))
    n = sum(grams.values())
    return -sum((c / n) * math.log2(c / n) for c in grams.values())


def top_kgram_share(sequence: Sequence[str], k: int) -> float:
    """Share of positions taken by the single most common k-gram."""
    if k < 1 or len(sequence) < k:
        return 0.0
    grams = Counter(tuple(sequence[i:i + k]) for i in range(len(sequence) - k + 1))
    return grams.most_common(1)[0][1] / sum(grams.values())


# Each statistic paired with whether a larger value means more structure. The
# orientation matters: entropies fall as organisation rises.
STATISTICS = {
    "transition_mutual_information": (transition_mutual_information, True),
    "next_state_predictability": (next_state_predictability, True),
    "conditional_entropy": (conditional_entropy, False),
}


def measure(sequence: Sequence[str], max_level: int = 4) -> Dict[str, float]:
    """Every statistic for one sequence, including per-level k-gram figures."""
    out = {name: fn(sequence) for name, (fn, _) in STATISTICS.items()}
    for k in range(2, max_level + 1):
        out[f"kgram_entropy_{k}"] = kgram_entropy(sequence, k)
        out[f"top_kgram_share_{k}"] = top_kgram_share(sequence, k)
    return out


def shuffle_null(
    sequence: Sequence[str],
    shuffles: int = 2000,
    max_level: int = 4,
    seed: Optional[int] = 0,
) -> Dict[str, Any]:
    """
    Compare the real ordering against order-destroying shuffles.

    The multiset of states is preserved exactly; only arrangement changes. So a
    difference here is temporal structure and nothing else, which is the whole
    point of the test.

    Returns each statistic with its real value, the null mean and standard
    deviation, a z-score, and a one-sided empirical p-value in the direction that
    means "more structured".
    """
    sequence = list(sequence)
    if len(sequence) < 3:
        return {"success": False, "error": "need at least 3 observations to test ordering"}

    rng = np.random.default_rng(seed)
    real = measure(sequence, max_level=max_level)

    null_samples: Dict[str, List[float]] = {name: [] for name in real}
    shuffled = list(sequence)
    for _ in range(shuffles):
        rng.shuffle(shuffled)
        for name, value in measure(shuffled, max_level=max_level).items():
            null_samples[name].append(value)

    results: Dict[str, Any] = {}
    for name, real_value in real.items():
        samples = np.asarray(null_samples[name], dtype=float)
        mean = float(samples.mean())
        std = float(samples.std(ddof=1)) if len(samples) > 1 else 0.0
        higher_is_structure = _orientation(name)

        if higher_is_structure:
            exceed = int((samples >= real_value).sum())
            z = (real_value - mean) / std if std > 0 else 0.0
        else:
            exceed = int((samples <= real_value).sum())
            z = (mean - real_value) / std if std > 0 else 0.0

        results[name] = {
            "real": round(real_value, 6),
            "null_mean": round(mean, 6),
            "null_std": round(std, 6),
            "z": round(z, 3),
            # Add-one bound: with n shuffles nothing smaller than 1/(n+1) has been
            # demonstrated, so p is never reported as zero.
            "p": round((exceed + 1) / (shuffles + 1), 6),
            "higher_is_structure": higher_is_structure,
        }

    return {
        "success": True,
        "observations": len(sequence),
        "distinct_states": len(set(sequence)),
        "shuffles": shuffles,
        "p_floor": round(1 / (shuffles + 1), 6),
        "statistics": results,
        "emergence": emergence_profile(results, max_level=max_level),
    }


def _orientation(name: str) -> bool:
    """True when a larger value of this statistic means more structure."""
    if name in STATISTICS:
        return STATISTICS[name][1]
    if name.startswith("kgram_entropy"):
        return False
    return True  # top_kgram_share: a stronger dominant motif is more structure


def emergence_profile(results: Dict[str, Any], max_level: int = 4,
                      alpha: float = 0.05) -> Dict[str, Any]:
    """
    E_k per level, with a verdict on whether the level earned promotion.

    Level 1 is the pairwise transition structure; levels 2 and up come from the
    k-gram statistics.

    Two rules keep this from promoting noise, both added after a control stream
    with real clusters but randomised order promoted E2 at p = 0.022:

    Testing several levels at once inflates false positives, so the threshold is
    Bonferroni-corrected by the number of levels tested and the effect must also
    be visible (|z| >= 3), because a large sample makes a negligible excess
    statistically significant.

    A level is only promoted if every level beneath it was. Structure has to earn
    its way up from the bottom; a fourth-order motif on top of a sequence with no
    first-order dependency is an artefact, not a hierarchy.
    """
    profile: Dict[str, Any] = {}
    levels = 1 + max(0, max_level - 1)
    corrected_alpha = alpha / max(1, levels)

    pairwise = results.get("transition_mutual_information", {})
    profile["E1"] = _level_entry(
        pairwise.get("real", 0.0) - pairwise.get("null_mean", 0.0),
        pairwise.get("z", 0.0), pairwise.get("p", 1.0),
        "transition_mutual_information", corrected_alpha)

    for k in range(2, max_level + 1):
        entropy = results.get(f"kgram_entropy_{k}")
        if not entropy:
            continue
        # Negated: organisation shows up as entropy below the null.
        profile[f"E{k}"] = _level_entry(
            entropy["null_mean"] - entropy["real"], entropy["z"], entropy["p"],
            f"kgram_entropy_{k}", corrected_alpha)

    # Contiguity from the bottom.
    highest = 0
    for k in range(1, max_level + 1):
        entry = profile.get(f"E{k}")
        if entry is None:
            break
        if entry["status"] != "candidate":
            break
        highest = k
    for k in range(highest + 1, max_level + 1):
        entry = profile.get(f"E{k}")
        if entry and entry["status"] == "candidate":
            entry["status"] = "unsupported-by-lower-level"

    profile["highest_promoted_level"] = highest
    profile["alpha"] = alpha
    profile["corrected_alpha"] = round(corrected_alpha, 6)
    return profile


def _level_entry(excess: float, z: float, p: float, source: str,
                 corrected_alpha: float) -> Dict[str, Any]:
    status = ("candidate" if (p <= corrected_alpha and abs(z) >= 3.0 and excess > 0)
              else "not-supported")
    return {
        "excess": round(excess, 6),
        "z": z,
        "p": p,
        "source": source,
        "status": status,
    }
