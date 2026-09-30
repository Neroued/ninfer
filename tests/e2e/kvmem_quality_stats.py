"""Conservative paired binary-accuracy intervals, using only the stdlib.

An exact binomial interval bounds each discordant probability. Bonferroni gives
joint coverage >= confidence; subtracting the bounds covers the paired accuracy
difference. This deliberately remains non-degenerate when no disagreements occur.
It measures sampling uncertainty, not model/quantization comparability.
"""
import math
from collections import defaultdict


def binomial_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n or p == 0:
        return 1.0
    if p == 1:
        return 0.0
    return min(1.0, math.fsum(math.exp(
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        + i * math.log(p) + (n - i) * math.log1p(-p)) for i in range(k + 1)))


def cdf_root(k, n, target):
    lo, hi = 0.0, 1.0
    for _ in range(64):
        mid = (lo + hi) / 2
        if binomial_cdf(k, n, mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def binomial_bounds(k, n, tail):
    return (0.0 if k == 0 else cdf_root(k - 1, n, 1 - tail),
            1.0 if k == n else cdf_root(k, n, tail))


def paired_accuracy(reference, candidate, confidence=0.95):
    if len(reference) != len(candidate) or not reference:
        raise ValueError("nonempty aligned pairs required")
    if not 0 < confidence < 1 or any(type(v) is not bool for v in reference + candidate):
        raise ValueError("boolean labels and valid confidence required")
    n = len(reference)
    better = sum(b and not a for a, b in zip(reference, candidate))
    worse = sum(a and not b for a, b in zip(reference, candidate))
    tail = (1 - confidence) / 4  # four one-sided bounds, union bound
    better_lo, better_hi = binomial_bounds(better, n, tail)
    worse_lo, worse_hi = binomial_bounds(worse, n, tail)
    return {"n": n, "reference_correct": sum(reference), "candidate_correct": sum(candidate),
            "improvements": better, "regressions": worse,
            "difference": (better - worse) / n, "confidence": confidence,
            "difference_interval": [better_lo - worse_hi, better_hi - worse_lo],
            "method": "paired discordance, exact Clopper-Pearson + Bonferroni",
            "equivalence_established": False}


def paired_cluster_accuracy(reference, candidate, cluster_ids, confidence=0.95):
    """Question-weighted contrast with arbitrary dependence inside each cluster.

For a size-s cluster let D=sum(candidate-reference), an integer in [-s,s].
E[D] = sum_{j=1}^s P(D>=j) - sum_{j=1}^s P(D<=-j). Within each size
stratum, exact binomial intervals bound these tail probabilities. Bonferroni
across all tails and strata gives simultaneous >= confidence coverage. Multiplying
by cluster_count / question_count retains the original question-weighted estimand.

Requires independent clusters and a common outcome distribution within each size
stratum. It makes no independence assumption inside a cluster. Small strata give
wide intervals; do not replace them with zero-width bootstrap ties or relax the
acceptance margin. Residual unmodelled dependence remains a limitation.
"""
    result = paired_accuracy(reference, candidate, confidence)
    if len(cluster_ids) != len(reference) or any(not isinstance(c, str) or not c for c in cluster_ids):
        raise ValueError('one nonempty cluster ID required for each paired question')
    groups = defaultdict(list)
    for a, b, group in zip(reference, candidate, cluster_ids):
        groups[group].append(int(b) - int(a))
    strata = defaultdict(list)
    for differences in groups.values():
        strata[len(differences)].append(sum(differences))
    # Each size-s stratum contributes 2*s probabilities, each with two bounds.
    tail = (1 - confidence) / (4 * sum(strata))
    lo, hi = 0.0, 0.0
    detail = []
    for size, differences in sorted(strata.items()):
        count = len(differences)
        stratum_lo, stratum_hi = 0.0, 0.0
        for threshold in range(1, size + 1):
            pos = sum(d >= threshold for d in differences)
            neg = sum(d <= -threshold for d in differences)
            pos_lo, pos_hi = binomial_bounds(pos, count, tail)
            neg_lo, neg_hi = binomial_bounds(neg, count, tail)
            stratum_lo += pos_lo - neg_hi
            stratum_hi += pos_hi - neg_lo
        weight = count / len(reference)
        lo += weight * stratum_lo
        hi += weight * stratum_hi
        detail.append({'cluster_size': size, 'clusters': count,
                       'net_question_improvements': sum(differences)})
    result.update(clusters=len(groups), strata=detail,
        difference_interval=[max(-1.0, lo), min(1.0, hi)],
        method='cluster-size strata, cumulative discordance tails, exact Clopper-Pearson + Bonferroni',
        sampling_assumption='independent clusters, common distribution within each cluster-size stratum')
    return result
