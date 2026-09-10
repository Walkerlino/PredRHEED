from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decisions import DecisionContractError, summarize_decisions


@dataclass(frozen=True, slots=True)
class BootstrapInterval:

    point: float
    lower: float
    upper: float
    valid_replicates: int


@dataclass(frozen=True, slots=True)
class LeaveOneClusterOutSummary:

    cluster_ids: tuple[object, ...]
    accuracies: tuple[float, ...]
    minimum: float
    maximum: float


LeaveOneSegmentOutSummary = LeaveOneClusterOutSummary


def _flatten_inputs(
    predicted_labels: np.ndarray,
    reference_labels: np.ndarray,
    cluster_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted_labels)
    reference = np.asarray(reference_labels)
    clusters = np.asarray(cluster_ids)
    if predicted.shape != reference.shape:
        raise DecisionContractError("predicted and reference labels must have identical shapes")
    if clusters.shape == predicted.shape:
        expanded_clusters = clusters
    elif predicted.ndim >= 1 and clusters.shape == predicted.shape[:1]:
        expanded_clusters = np.broadcast_to(
            clusters.reshape((-1,) + (1,) * (predicted.ndim - 1)),
            predicted.shape,
        )
    else:
        raise DecisionContractError("cluster_ids must identify every decision or every first-axis sample")
    flattened_clusters = expanded_clusters.reshape(-1)
    if flattened_clusters.dtype.kind in {"f", "c"}:
        has_nonfinite = not np.isfinite(flattened_clusters).all()
    elif flattened_clusters.dtype.kind == "O":
        has_nonfinite = any(
            isinstance(value, (float, complex, np.floating, np.complexfloating))
            and not np.isfinite(value)
            for value in flattened_clusters.tolist()
        )
    else:
        has_nonfinite = False
    if has_nonfinite:
        raise DecisionContractError("cluster_ids must contain only finite identifiers")
    return predicted.reshape(-1), reference.reshape(-1), flattened_clusters


def cluster_bootstrap_accuracy(
    predicted_labels: np.ndarray,
    reference_labels: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    replicates: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 2025,
) -> BootstrapInterval:

    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise DecisionContractError("replicates must be a positive integer")
    if not 0.0 < confidence_level < 1.0:
        raise DecisionContractError("confidence_level must lie strictly between zero and one")
    predicted, reference, clusters = _flatten_inputs(predicted_labels, reference_labels, cluster_ids)
    unique_clusters = np.unique(clusters)
    if unique_clusters.size < 2:
        raise DecisionContractError("cluster bootstrap requires at least two clusters")

    point = summarize_decisions(predicted, reference).overall_accuracy
    if not np.isfinite(point):
        raise DecisionContractError("at least one target-second decision is required")
    indices_by_cluster = [np.flatnonzero(clusters == cluster) for cluster in unique_clusters]
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(replicates):
        selected = rng.integers(0, unique_clusters.size, size=unique_clusters.size)
        sample_indices = np.concatenate([indices_by_cluster[index] for index in selected])
        accuracy = summarize_decisions(
            predicted[sample_indices], reference[sample_indices]
        ).overall_accuracy
        if np.isfinite(accuracy):
            samples.append(accuracy)
    if not samples:
        raise DecisionContractError("no bootstrap replicate contained a target-second decision")

    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(np.asarray(samples), [tail, 1.0 - tail])
    return BootstrapInterval(
        point=point,
        lower=float(lower),
        upper=float(upper),
        valid_replicates=len(samples),
    )


def leave_one_cluster_out_accuracy(
    predicted_labels: np.ndarray,
    reference_labels: np.ndarray,
    cluster_ids: np.ndarray,
) -> LeaveOneClusterOutSummary:

    predicted, reference, clusters = _flatten_inputs(predicted_labels, reference_labels, cluster_ids)
    unique_clusters = np.unique(clusters)
    if unique_clusters.size < 2:
        raise DecisionContractError("leave-one-cluster-out requires at least two clusters")
    accuracies: list[float] = []
    for cluster in unique_clusters:
        keep = clusters != cluster
        accuracy = summarize_decisions(predicted[keep], reference[keep]).overall_accuracy
        if not np.isfinite(accuracy):
            raise DecisionContractError("each leave-one-cluster-out subset must contain a decision")
        accuracies.append(accuracy)
    return LeaveOneClusterOutSummary(
        cluster_ids=tuple(unique_clusters.tolist()),
        accuracies=tuple(float(value) for value in accuracies),
        minimum=float(np.min(accuracies)),
        maximum=float(np.max(accuracies)),
    )


def leave_one_segment_out_accuracy(
    predicted_labels: np.ndarray,
    reference_labels: np.ndarray,
    segment_ids: np.ndarray,
) -> LeaveOneSegmentOutSummary:

    return leave_one_cluster_out_accuracy(predicted_labels, reference_labels, segment_ids)
