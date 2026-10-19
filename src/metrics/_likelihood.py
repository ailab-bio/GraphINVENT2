"""
Likelihood metrics: the negative log-likelihood of a sample under the model,
and the uniformity-completeness Jensen-Shannon divergence.

These were the last metrics computed outside `src/metrics/`. The computation
lived in `Analyzer`, which made it the one exception to the rule that a metric
has exactly one implementation and that the implementation is here. The model
forward pass still belongs to `Analyzer`, because only it holds the model and
the data loaders; what moved is the arithmetic on the likelihood tensors, which
is what the metric actually is.
"""

from __future__ import annotations


def uc_jsd(
    likelihood_valid,
    likelihood_train,
    likelihood_sampled,
) -> float:
    """
    `uc_jsd`: the uniformity-completeness Jensen-Shannon divergence between the
    per-action negative log-likelihoods of the validation set, the training set
    and a generated sample.

    From Arús-Pous et al., *J. Chem. Inf. Model.* 59:1 (2019). A model that has
    learned the distribution rather than memorized it puts similar likelihood
    on held-out molecules and on its own samples, so the three distributions
    agree and the divergence is small.

    The three tensors are truncated to their shortest length before being
    normalized, because they come from different sample sizes and the
    divergence is defined between distributions of equal support.
    """
    import torch

    min_len = min(len(likelihood_valid), len(likelihood_sampled), len(likelihood_train))
    if min_len == 0:
        raise ValueError(
            "uc_jsd needs non-empty likelihoods for the validation set, the "
            "training set and a generated sample; one of the three is empty."
        )

    def normalize(values):
        head = values[:min_len]
        total = torch.sum(head)
        return head / total

    valid = normalize(likelihood_valid)
    train = normalize(likelihood_train)
    sampled = normalize(likelihood_sampled)

    mixture = (valid + train + sampled) / 3
    log_mixture = torch.log(mixture + 1e-8)
    divergence = (
        torch.nn.functional.kl_div(log_mixture, valid, reduction="sum")
        + torch.nn.functional.kl_div(log_mixture, train, reduction="sum")
        + torch.nn.functional.kl_div(log_mixture, sampled, reduction="sum")
    ) / 3
    return float(divergence)


def nll_summary(train, validation, test=None) -> dict:
    """
    `nll_train`, `nll_validation` and `nll_test`.

    Mean negative log-likelihood per molecule, summed over construction
    actions. Comparable across the splits of one dataset and not across
    datasets with different molecule sizes, because a longer construction route
    accumulates more terms.
    """
    return {
        "nll_train": float(train),
        "nll_validation": float(validation),
        "nll_test": None if test is None else float(test),
    }
