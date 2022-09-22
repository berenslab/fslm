from __future__ import annotations

from copy import deepcopy
from typing import Generator, List, Optional, Tuple

import torch
from nflows.flows import Flow
from pyknos.mdn.mdn import MultivariateGaussianMDN as mdn
from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior
from sbi.inference.posteriors.rejection_posterior import RejectionPosterior

# types
from torch import Tensor

from fslm.utils import extract_and_transform_mog


def build_reducable_posterior(inference_obj, **kwargs):
    posterior = inference_obj.build_posterior(**kwargs)
    return ReducablePosterior(posterior)


def ReducablePosterior(posterior: RejectionPosterior or MCMCPosterior):
    r"""Factory function to wrap `MCMCPosterior` or `RejectionPosterior`.

    Provides passthrough to wrap the posterior instance in its respective
    reducable class wrapper `ReducableRejectionPosterior` or
    `ReducableMCMCPosterior` depending on the instance's class.

    Example:
        ```
        posterior = infer(simulator, prior, "SNLE_A", num_simulations)
        reducable_posterior = ReducablePosterior(posterior)
        reducable_posterior.marginalise_likelihood(list_of_dims_to_keep)
        reducable_posterior.sample()
        ```
    Args:
        posterior: sbi `MCMCPosterior` or `RejectionPosterior` instance that has
            been trained using an MDN.

    Returns:
        ReducableRejectionPosterior` or `ReducableMCMCPosterior`
    """
    if isinstance(posterior, RejectionPosterior):
        return ReducableRejectionPosterior(deepcopy(posterior))
    elif isinstance(posterior, MCMCPosterior):
        return ReducableMCMCPosterior(deepcopy(posterior))


class ReducableBasePosterior:
    r"""Provides marginalisation functionality to for `MCMCPosterior` and
    `RejectionPosterior`.


    Args:
        posterior: sbi `MCMCPosterior` or `RejectionPosterior` instance that has
            been trained using an MDN.

    Attributes:
        Inherits all its attributes from posterior instance.
    """

    def __init__(self, posterior: MCMCPosterior or RejectionPosterior) -> None:
        self._wrapped_posterior = posterior

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._wrapped_posterior, attr)

    def marginalise_likelihood(
        self, dims: List[int], inplace: bool = True
    ) -> Optional[ReducablePosterior]:
        r"""Marginalise likelihood distribution of the likelihood-based posterior.

        Marginalises the MDN-based likelihood $p(x_1, ..., x_N|\theta)$ such that
        $$
        p(\theta|x_{subset}) \propto p(x_{subset}|\theta) p(\theta)
        $$
        , where $x_{susbet} \susbet (x_1, ..., x_N)$

        Args:
            dims: Feature dimensions to keep.
            inplace: Whether to return a marginalised copy of self, or to
                marginalise self directly.

        Returns:
            red_posterior: If inplace=False, returns a marginalised copy of self.
        """
        likelihood_estimator = self.potential_fn.likelihood_estimator
        marginal_likelihood = ReducableLikelihoodEstimator(likelihood_estimator, dims)
        if inplace:
            self.potential_fn.likelihood_estimator = marginal_likelihood
        else:
            red_posterior = deepcopy(self)
            red_posterior.potential_fn.likelihood_estimator = marginal_likelihood
            return red_posterior


class ReducableRejectionPosterior(ReducableBasePosterior, RejectionPosterior):
    r"""Wrapper for `RejectionPosterior` that make use of a
    MDN as their density estimator. Implements functionality to evaluate
    and sample the posterior based on a reduced subset of features p(\theta|x1)
    of x_o = (x1, x2).

    Args:
        posterior: `RejectionPosterior` instance trained using an MDN.

    Attributes:
        Inherits all its attributes from posterior instance.
    """

    def __init__(self, posterior: RejectionPosterior) -> None:
        self._wrapped_posterior = posterior


class ReducableMCMCPosterior(ReducableBasePosterior, MCMCPosterior):
    r"""Wrapper for `MCMCPosterior` that make use of a
    MDN as their density estimator. Implements functionality to evaluate
    and sample the posterior based on a reduced subset of features $p(\theta|x1)$
    of $x_o = (x1, x2)$.

    Args:
        posterior: `MCMCPosterior` instance that trained using an MDN.

    Attributes:
        Inherits all its attributes from posterior instance.
    """

    def __init__(self, posterior: MCMCPosterior) -> None:
        self._wrapped_posterior = posterior


class ReducableLikelihoodEstimator:
    r"""Adds marginalisation functionality to mdn based likelihood estimators.

    Supports `.log_prob` of the MoG likelihood.

    Its main purpose is to emulate the likelihood estimator employed in
    `LikelihoodbasedPotential`.

    Args:
        likelihood_estimator: Flow instance that was trained using a MDN.
        marginal_dims: List of x dimensions to consider. Dimensions not
            in `marginal_dims` are marginalised out.

    Attributes:
        likelihood_net: Conditional density estimator for the likelihood.
        dims: List of x dimensions to consider in the evaluation.
    """

    def __init__(
        self, likelihood_estimator: Flow, marginal_dims: Optional[List[int]] = None
    ) -> None:
        self.likelihood_net = likelihood_estimator
        self.dims = marginal_dims

    def parameters(self) -> Generator:
        """Provides pass-through to `parameters()` of self.likelihood_net.

        Used for infering device.

        Returns:
            Generator for the model parameters.
        """
        return self.likelihood_net.parameters()

    def eval(self) -> Flow:
        """Provides pass-through to `eval()` of self.likelihood_net.

        Returns:
            Flow model.
        """
        return self.likelihood_net.eval()

    def marginalise_likelihood(
        self, context: Tensor, dims: List[int]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Marginalise MoG and return new mixture parameters.

        Args:
            context: Condition in $p(x|\theta)$.
            dims: List of dimensions to keep.

        Returns:
            logits: log-weights for each component of the marginal distributions.
            mu_x: means of the marginal distributution for each component.
            precfs_xx: precision factors of the marginal distribution for
                each component.
            sumlogdiag: Sum of the logarithms of the diagonal elements of the
                precision factors of the marginal distributions for each component.
        """

        # reset to unmarginalised params
        logits, means, precfs, _ = extract_and_transform_mog(
            self.likelihood_net, context
        )

        mask = torch.zeros(means.shape[-1], dtype=bool)
        mask[dims] = True

        # Make a new precisions with correct ordering
        mu_x = means[:, :, mask]
        precfs_xx = precfs[:, :, mask]
        precfs_xx = precfs_xx[:, :, :, mask]

        # set new GMM parameters
        sumlogdiag = torch.sum(
            torch.log(torch.diagonal(precfs_xx, dim1=2, dim2=3)), dim=2
        )
        return logits, mu_x, precfs_xx, sumlogdiag

    def log_prob(self, inputs: Tensor, context: Tensor) -> Tensor:
        """Evaluate the Mixture of Gaussian (MoG)
        probability density function at a value x.

        Args:
            inputs: Values at which to evaluate the MoG pdf.
            context: Conditiones the likelihood distribution.

        Returns:
            Log probabilities at values specified by theta.
        """
        logits, means, precfs, sumlogdiag = self.marginalise_likelihood(context, self.dims)
        prec = precfs.transpose(3, 2) @ precfs

        return mdn.log_prob_mog(inputs[:, self.dims], logits, means, prec, sumlogdiag)
