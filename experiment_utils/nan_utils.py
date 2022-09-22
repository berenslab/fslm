from __future__ import annotations

import torch
from torch import Tensor
from copy import deepcopy
import torch.nn as nn
from nflows.flows import Flow
import warnings
from typing import Any, Callable, Dict, Generator, Optional, Tuple

from torch.distributions import Distribution, constraints
from torch.utils.data import DataLoader, TensorDataset


def includes_nan(X: Tensor) -> Tensor:
    """Checks if obsevation contains NaNs.

    Args:
        X: Batch of observations x_i, (batch_size, num_features).

    Returns:
        True if x_i contains NaN feature.
    """
    has_inf = torch.any(X.isinf(), axis=1)
    has_nan = torch.any(X.isnan(), axis=1)
    return torch.logical_or(has_inf, has_nan)

class WillItSimulate(nn.Module):
    r"""Classifier to predict whether a simulator setting $\theta$ will produce
    observations x that contain NaNs.

    The model is optimises a Log

    Attributes:
        fc1: ipnut layer
        fc2: output layer
        relu: ReLU activation
        softmax: Logistic Softmax

        device: Training device.
        n_params: Dimenisionality of prior.
    """

    def __init__(self, num_params: int, num_hidden: int = 50, device: str = "cpu"):
        """
        Args:
            num_params: The number of input parameters of the simulator model.
            num_hidden: The number of hidden units per layer.
        """
        super(WillItSimulate, self).__init__()
        self.fc1 = nn.Linear(num_params, num_hidden, device=device)
        self.fc2 = nn.Linear(num_hidden, num_hidden, device=device)
        self.fc3 = nn.Linear(num_hidden, 2, device=device)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

        self.device = device
        self.n_params = num_params

    def forward(self, theta: Tensor) -> Tensor:
        """Implements the forward pass.

        Args:
            theta: Simulator parameters, (batch_size, num_params).

        Returns:
            p_nan: log_probs for will produce no NaNs [0] vs NaNs [1].
        """
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = self.relu(theta)
        theta = self.fc3(theta)
        p_nan = self.softmax(theta)
        return p_nan

    def log_prob(self, theta: Tensor, y: int = 0) -> Tensor:
        r"""Likelihood function of the model.

        $p(\theta|y)$, where y=1 means $\theta$ will produce no NaNs.

        Args:
            theta: Simulator parameters, (batch_size, num_params),
            y: Produces no NaNs = 0. Produces NaNs = 1.

        Returns:
            p_nan: log_probs for will produce NaNs [0] vs no NaNs [1].
        """
        probs = self.forward(theta.view(-1, self.n_params))
        p_nan = probs[:, y]
        return p_nan

    def train(
        self,
        theta_train: Tensor,
        y_train: Tensor,
        batch_size: int = 1000,
        lr: float = 1e-3,
        max_epochs: int = 10000,
        val_ratio=0.1,
        stop_after_epoch: int = 50,
        verbose: bool = True,
    ):
        r"""Trains classifier to predict parameters that are likely to cause
        NaN observations.

        Returns self for calls such as:
        `nan_likelihood = WillItSimulate(n_params).train(theta, y)`

        Args:
            theta_train: Sets of parameters for training.
            y_train: Observation labels. 0 if x_i contains no NaNs, 1 if does.
            lr: Learning rate.
            max_epochs: Number of epochs to train for.
            stop_after_epoch: Stop after number of epochs without validation
                improvement.
            val_ratio: How any training_examples to split of for validation.
            batch_size: Training batch size.

        Returns:
            self
        """
        # TODO: integrate tensorboard or progress report and loss etc

        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        n_split = int(len(theta_train) * val_ratio)

        train_data = TensorDataset(theta_train[n_split:], y_train[n_split:])
        val_data = TensorDataset(theta_train[:n_split], y_train[:n_split])

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

        best_val_log_prob = 0
        epochs_without_improvement = 0
        for epoch in range(max_epochs):
            train_log_probs_sum = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                theta_batch, y_batch = (
                    batch[0].to(self.device),
                    batch[1].to(self.device),
                )
                train_losses = criterion(self.forward(theta_batch), y_batch)
                train_loss = torch.mean(train_losses)
                train_log_probs_sum -= train_losses.sum().item()

                train_loss.backward()
                optimizer.step()

            train_log_prob_average = train_log_probs_sum / (
                len(train_loader) * train_loader.batch_size
            )

            val_log_probs_sum = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, y_batch = (
                        batch[0].to(self.device),
                        batch[1].to(self.device),
                    )
                    val_losses = criterion(self.forward(theta_batch), y_batch)
                    val_log_probs_sum -= val_losses.sum().item()

            val_log_prob_average = val_log_probs_sum / (
                len(val_loader) * val_loader.batch_size
            )

            if epoch == 0 or val_log_prob_average > best_val_log_prob:
                best_val_log_prob = val_log_prob_average
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement > stop_after_epoch:
                print(f"converged after {epoch} epochs.")
                break

            if verbose:
                print(
                    "\r[{}] train loss: {:.5f}, val_loss: {:.5f}".format(
                        epoch, train_log_prob_average, val_log_prob_average
                    ),
                    end="",
                )

        return self


class OptimisedPrior(Distribution):
    r"""Prior distribution that can be optimised to cover less parameters
    $\theta$, that produce observations x containing NaNs.

    For a given prior distribution $p(\theta)$, that we sample to obtain
    parameters $\theta$, we can construct a new modified prior according to:
    $$
    \tilde{p}(\theta)=p(\theta|y=1)=\frac{p(y=0|\theta)p(\theta)}{p(y)}
    $$
    , where p(y) is the probability of $\theta$ producing non valid observations
    (y=1) and $p(y|\theta)$ is a binary probabilistic classifier that predicts
    whether a set of parameters $\theta$ is likely to produce observations that
    include non valid (NaN or inf) features.

    It has `.sample()` and `.log_prob()` methods. Sampling is realised through
    rejection sampling. The support is inherited from the initial prior
    distribution.

    Args:
        prior: A base_prior distribution.
        device: Which device to use.

    Attributes:
        base_prior: The prior distribution that is optimised.
        dim: The dimensionality of $\theta$.
        nan_likelihood: Classifier to predict if $\theta$ will produce NaNs in
            observation.
    """

    def __init__(self, prior: Any, device: str = "cpu"):
        r"""
        Args:
            prior: A prior distribution that supports `.log_prob()` and
                `.sample()`.
            device: which device to use. Should be same as for `prior`.
        """
        self.base_prior = prior
        self.dim = prior.sample((1,)).shape[1]
        self.nan_likelihood = None
        self.device = device
        self._mean = prior.mean
        self._variance = prior.variance

    @property
    def mean(self):
        if self.nan_likelihood is None:
            return self.base_prior.mean
        else:
            return self._mean

    @property
    def variance(self):
        if self.nan_likelihood is None:
            return self.base_prior.variance
        else:
            return self._variance

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        return self.base_prior.arg_constraints

    @property
    def support(self) -> constraints.Constraint:
        return self.base_prior.support

    def sample(self, sample_shape: Tuple, y: int = 0) -> Tensor:
        """Sample the optimised prior distribution.

        Samples the optimised prior via rejection sampling in the support of the
        original prior.

        Args:
            sample_shape: Shape of the sample. (n_batches, batch_size)
            y: Whether the samples will produce NaNs or not. Default is 0,
                since the desired behaviour is to supress NaN features.

        Returns:
            Samples from the optimised prior.
        """
        if self.nan_likelihood is None:
            warnings.warn(
                "Sampling non optimised prior! To optimise, call .train() first!"
            )
            return self.base_prior.sample(sample_shape)
        else:
            n_samples = torch.Size(sample_shape).numel()
            n = 0
            samples = []
            # rejection sampling | could be replaced by sbi's rejection_sample
            # return rejection_sample(self, self.base_prior, num_samples)

            while n < n_samples:
                theta = self.base_prior.sample((n_samples - n,))
                acceptance_probs = torch.exp(self.nan_likelihood(theta)[:, y])
                accepted = acceptance_probs > torch.rand_like(acceptance_probs)
                samples.append(theta[accepted])
                n = len(torch.vstack(samples))
            return torch.vstack(samples).reshape((*sample_shape, -1))

    def log_prob(self, theta: Tensor, y: int = 0) -> Tensor:
        """Prob whether theta will produce NaN observations under prior.

        Args:
            theta: Simulator parameters, (batch_size, num_params),
            y: Produces no NaNs = 0. Produces NaNs = 1.

        Returns:
            p_nan: log_probs for will produce no NaNs [0] vs NaNs [1].
        """
        if self.nan_likelihood is None:
            warnings.warn(
                "Evaluating non optimised prior! To optimise, call .train() first!"
            )
            return self.base_prior.log_prob(theta)
        else:
            with torch.no_grad():
                p_no_nan = self.nan_likelihood(theta.view(-1, self.dim))[:, y]
                p = self.base_prior.log_prob(theta)
                Z = self.Z(y)
                return (p_no_nan + p - Z).squeeze()

    def Z(self, y: int) -> Tensor:
        """Normalisation constant p(NaN).

        Args:
            y: Produces no NaNs = 0. Produces NaNs = 1.

        Returns:
            Marginal log_prob of producing NaNs.
        """
        return self._norm[y]

    def train(
        self,
        theta_train: Tensor,
        x_train: Tensor = None,
        y_train: Tensor[bool] = None,
        lr: float = 1e-3,
        max_epochs: int = 10000,
        batch_size: int = 1000,
        val_ratio=0.1,
        stop_after_epoch: int = 50,
        verbose: bool = True,
    ) -> OptimisedPrior:
        r"""Trains a classifier to predict parameters that are likely to cause
        NaN observations.

        Returns self for calls such as:
        `trained_prior = OptimisedPrior(prior).train(theta, x)`

        Args:
            theta_train: Sets of parameters for training.
            x_train: Set of training observations that some of which include
                NaN features. Will be used to create labels y_train, depending
                on presence of NaNs (True if x_i contains NaN, else False).
            y_train: Labels whether corresponding theta_train produced an
                observation x_train that included NaN features.
            lr: Learning rate.
            max_epochs: Number of epochs to train for.
            batch_size: Training batch size.
            stop_after_epoch: Stop after number of epochs without validation
                improvement.
            val_ratio: How any training_examples to split of for validation.

        Returns:
            self trained optimised prior.
        """
        if x_train is not None:
            has_nan = includes_nan(x_train)
        elif y_train is not None:
            has_nan = y_train
        else:
            raise ValueError("Please provide y_train or x_train.")
        no_nans_vs_nans = torch.vstack([~has_nan, has_nan])
        Z = torch.log(no_nans_vs_nans.count_nonzero(axis=1) / len(x_train))
        self._norm = Z  # lambda function for norm / Z provokes pickling error

        self.nan_likelihood = WillItSimulate(
            num_params=self.dim, device=self.device
        ).train(
            theta_train,
            has_nan.long(),
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            verbose=verbose,
            val_ratio=val_ratio,
            stop_after_epoch=stop_after_epoch,
        )

        # estimating mean and variance from samples
        samples = self.sample((10000,))
        self._mean = samples.mean()
        self._variance = samples.var()

        return self

def optimise_prior(
    prior: Any, simulator: Callable, num_samples: int = 1000
) -> OptimisedPrior:
    r"""Optimise prior to include less samples that produce NaN observations.

    Args:
        prior: Any parameter prior supported by `sbi`.
        simulator: Function that takes parameter tensor $\theta$ of shape
            (num_samples, num_params) and produces output $x$ of shape
            (num_samples, num_features)
        num_samples: How many samples to train the optimised prior on.

    Returns:
        OptimisedPrior instance, trained on samples from the provided simulator.
    """
    theta_train = prior.sample((num_samples,))
    x_train = simulator(theta_train)
    return OptimisedPrior(prior).train(theta_train, x_train)


class NaNCalibration(nn.Module):
    r"""Learns calibration bias to compensate for NaN observations.

    Logistic regressor predicts which parameters cause NaN observations.
    The model is optimised with a binary cross entropy loss.

    The forward pass computes $p(valid|\theta)$.

    For reference (https://openreview.net/pdf?id=kZ0UYdhqkNY)

    Args:
        input_dim: Number of parameters in $\theta$.
        device: On which device to train.

    Attributes:
        linear: Linear layer.
        device: Which device is used.
    """

    def __init__(self, input_dim: int, device: str = "cpu"):
        super(NaNCalibration, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
        self.device = device

    def forward(self, inputs: Tensor) -> Tensor:
        r"""Implements logistic regression

        Args:
            inputs: theta.

        Returns:
            outputs: outputs of the forward pass. $p(valid|\theta)$
        """
        outputs = torch.sigmoid(self.linear(inputs))
        return outputs

    def log_prob(self, theta: Tensor) -> Tensor:
        r"""log of the forward pass, i.e. $p(valid|\theta)$.

        Args:
            theta: Simulator parameters, (batch_size, num_params),

        Returns:
            $log(p(valid|\theta))$.
        """
        probs = self.forward(theta.view(-1, self.n_params))
        return torch.log(probs)

    def train(
        self,
        theta_train: Tensor,
        y_train: Tensor[bool] = None,
        lr: float = 1e-3,
        batch_size: int = 1000,
        max_epochs: int = 5000,
        val_ratio=0.1,
        stop_after_epoch: int = 50,
        verbose: bool = True,
    ) -> NaNCalibration:
        r"""Trains classifier to predict the probability of valid observations.

        Returns self for calls such as:
        `nan_likelihood = NaNCallibration(n_params).train(theta, y)`

        Args:
            theta_train: Sets of parameters for training.
            y_train: Observation labels. 0 if x_i contains no NaNs, 1 if does.
            lr: Learning rate.
            num_epochs: Number of epochs to train for.
            batch_size: Training batch size.
            batch_size: Training batch size.
            stop_after_epoch: Stop after number of epochs without validation
                improvement.
            val_ratio: How any training_examples to split of for validation.

        Returns:
            self
        """
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        n_split = int(len(theta_train) * val_ratio)

        train_data = TensorDataset(theta_train[n_split:], y_train[n_split:])
        val_data = TensorDataset(theta_train[:n_split], y_train[:n_split])

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

        best_val_log_prob = 0
        epochs_without_improvement = 0
        for epoch in range(max_epochs):
            train_log_probs_sum = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                theta_batch, y_batch = (
                    batch[0].to(self.device),
                    batch[1].to(self.device),
                )
                train_losses = criterion(self.forward(theta_batch), y_batch)
                train_loss = torch.mean(train_losses)
                train_log_probs_sum -= train_losses.sum().item()

                train_loss.backward()
                optimizer.step()

                train_log_prob_average = train_log_probs_sum / (
                    len(train_loader) * train_loader.batch_size
                )

            val_log_probs_sum = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, y_batch = (
                        batch[0].to(self.device),
                        batch[1].to(self.device),
                    )
                    val_losses = criterion(self.forward(theta_batch), y_batch)
                    val_log_probs_sum -= val_losses.sum().item()

            val_log_prob_average = val_log_probs_sum / (
                len(val_loader) * val_loader.batch_size
            )

            if epoch == 0 or val_log_prob_average > best_val_log_prob:
                best_val_log_prob = val_log_prob_average
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement > stop_after_epoch:
                print(f"converged after {epoch} epochs.")
                break

            if verbose:
                print(
                    "\r[{}] train loss: {:.5f}, val_loss: {:.5f}".format(
                        epoch, train_log_prob_average, val_log_prob_average
                    ),
                    end="",
                )
        return self


class CalibratedPrior(Distribution):
    r"""Prior distribution that can be calibrated to compensate for the
    likelihood bias, that is caused by ignoring non-valid observations.

    Since the likelihood obtained through SNLE is biased according to
    $\ell_ψ(x_o |θ) \approx 1/Z p(x_o|θ)/p(valid|θ)$, we can account for this
    bias by estimating $c_ζ(θ)=p(valid|θ)$, such that:
    $$
    \ell_ψ(x_o|θ)\ell_ψ(x_o|θ)p(θ)c_ζ(θ) \propto p(x_o|θ)p(θ) \propto p(θ|x_o )
    $$

    For a given prior distribution $p(\theta)$, we can hence obtain a calibratedd
    prior $\tilde{p}(\theta)$ according to:
    $$
    \tilde{p}(\theta) \propto p(\theta)c_ζ(θ).
    $$
    Here $c_ζ(θ)$ is a logistic regressor that predicts whether a set of
    parameters $\theta$ produces valid observations features.

    Its has a modified `.log_prob()` method compared to the base_prior. While,
    sampling is passed on to the base prior.
    The support is inherited from the base prior distribution.


    Example:
    ```
    inference = SNLE_A(prior, show_progress_bars=True, density_estimator="mdn")
    estimator = inference.append_simulations(theta, x).train()
    calibrated_prior = CallibratedPrior(inference._prior).train(theta,x)
    posterior = inference.build_posterior(prior=calibrated_prior, sample_with='rejection')
    ```

    Args:
        prior: A base_prior distribution.
        device: Which device to use.

    Attributes:
        base_prior: The prior distribution that is optimised.
        dim: The dimensionality of $\theta$.
        nan_likelihood: Classifier to predict if $\theta$ will produce NaNs in
            observation.
    """

    def __init__(self, prior: Any, device: str = "cpu"):
        r"""
        Args:
            prior: A prior distribution that supports `.log_prob()` and
                `.sample()`.
            device: Which device to use. Should be same as for `prior`.
        """
        self.base_prior = prior
        self.dim = prior.sample((1,)).shape[1]
        self.nan_calibration = None
        self.device = device
        self._mean = prior.mean
        self._variance = prior.variance

    @property
    def mean(self):
        if self.nan_calibration is None:
            return self.base_prior.mean
        else:
            return self._mean

    @property
    def variance(self):
        if self.nan_calibration is None:
            return self.base_prior.variance
        else:
            return self._variance

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        return self.base_prior.arg_constraints

    @property
    def support(self) -> constraints.Constraint:
        return self.base_prior.support

    def log_prob(self, theta: Tensor) -> Tensor:
        """Prob of calibrated prior.

        Args:
            theta: Simulator parameters, (batch_size, num_params),

        Returns:
            p_nan: log_probs for $p(valid|\theta)$
        """
        if self.nan_calibration is None:
            warnings.warn(
                "Evaluating non calibrated prior! To calibrate, call .train() first!"
            )
            return self.base_prior.log_prob(theta)
        else:
            p_no_nan = self.nan_calibration(theta.view(-1, self.dim)).view(-1)
            p = self.base_prior.log_prob(theta)
            return p + p_no_nan

    def sample(self, *args, **kwargs):
        """Pass through to `self.base_prior.sample()`

        Returns:
            Samples from `base_prior`
        """
        return self.base_prior.sample(*args, **kwargs)

    def train(
        self,
        theta_train: Tensor,
        x_train: Tensor = None,
        y_train: Tensor[bool] = None,
        lr: float = 1e-3,
        batch_size: int = 1000,
        max_epochs: int = 5000,
        val_ratio=0.1,
        stop_after_epoch: int = 50,
        verbose: bool = True,
    ) -> CalibratedPrior:
        r"""Trains classifier to predict which parameters produce valid observations.

        Callibration factor is then added to log_prob of `base_prior`.

        The model is a logistic regressor optimised with a binary cross entropy
        loss.

        Returns self for calls such as:
        `trained_prior = CalibratedPrior(prior).train(theta, x)`

        Args:
            theta_train: Sets of parameters for training.
            x_train: Set of training observations that some of which include
                NaN features. Will be used to create labels y_train, depending
                on presence of NaNs (True if x_i contains NaN, else False).
            y_train: Labels whether corresponding theta_train produced an
                observation x_train that included NaN features.
            lr: Learning rate.
            num_epochs: Number of epochs to train for.
            batch_size: Training batch size.
            batch_size: Training batch size.
            stop_after_epoch: Stop after number of epochs without validation
                improvement.
            val_ratio: How any training_examples to split of for validation.

        Returns:
            self trained calibrated prior.
        """
        if x_train is not None:
            has_nan = includes_nan(x_train).view(-1, 1)
        elif y_train is not None:
            has_nan = y_train.view(-1, 1)
        else:
            raise ValueError("Please provide y_train or x_train.")

        self.nan_calibration = NaNCalibration(self.dim, device=self.device).train(
            theta_train,
            has_nan.float(),
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            val_ratio=val_ratio,
            stop_after_epoch=stop_after_epoch,
            verbose=verbose,
        )

        # estimating mean and variance from samples
        samples = self.sample((10000,))
        self._mean = samples.mean()
        self._variance = samples.var()

        return self


class CalibratedLikelihoodEstimator:
    r"""Modifies the likelihood by a calibration factor.

    Wraps the trained likelihood estimator and applies a calibration term to
    compensate for discarded training data due to NaN observations.

    Since the likelihood obtained through SNLE is biased according to
    $\ell_ψ(x_o |θ) \approx 1/Z p(x_o|θ)/p(valid|θ)$, we can account for this
    bias by estimating $c_ζ(θ)=p(valid|θ)$, such that:
    $$
    \ell_ψ(x_o|θ)\ell_ψ(x_o|θ)p(θ)c_ζ(θ) \propto p(x_o|θ)p(θ) \propto p(θ|x_o )
    $$

    For a given likelihood distribution $p(x|\theta)$, we can hence obtain a
    calibrated likelihood $\tilde{p}(x|\theta)$ according to:
    $$
    \tilde{p}(x|\theta) \propto p(x|\theta)c_ζ(θ).
    $$
    Here $c_ζ(θ)$ is a logistic regressor that predicts whether a set of
    parameters $\theta$ produces valid observations features.

    Example:
    ```
    inference = SNLE_A(prior, show_progress_bars=True, density_estimator="mdn")
    estimator = inference.append_simulations(theta, x).train()
    calibrated_likelihood = calibrate_likelihood_estimator(estimator, theta, x)
    posterior = inference.build_posterior(density_estimator=calibrated_likelihood, sample_with='rejection')
    ```

    Args:
        likelihood_estimator: A likelihood estimator (a Flow).
        calibration_f: A trained callibration network that has learned
            $p(valid|\theta)$.

    Attributes:
        calibration_f: calibration factor for likelihood that has been trained
        partially on NaNs.
    """

    def __init__(self, likelihood_estimator: Flow, calibration_f: NaNCalibration):
        self._wrapped_estimator = likelihood_estimator
        self.calibration_f = calibration_f

    def __getattr__(self, attr):
        """Forward attrs to wrapped object if not existant in self."""
        if attr == "log_prob":
            return getattr(self._wrapped_estimator, attr)
        elif attr in self.__dict__:
            return getattr(self, attr)

    def parameters(self) -> Generator:
        """Provides pass-through to `parameters()` of self.likelihood_net.

        Used for infering device.

        Returns:
            Generator for the model parameters.
        """
        return self._wrapped_estimator.parameters()

    def eval(self) -> Flow:
        """Provides pass-through to `eval()` of self.likelihood_net.

        Returns:
            Flow model.
        """
        return self._wrapped_estimator.eval()

    def log_prob(self, inputs: Tensor, context: Optional[Tensor] = None) -> Tensor:
        r"""calibrated likelihood.

        Adds the calibration log_prob $p(valid|\theta)$ on top of the likelihood
        log_prob.

        Args:
            inputs: where to evaluate the likelihood.
            context: context of the likelihood.

        Returns:
            calibrated likelihoods.
        """
        return self._wrapped_estimator.log_prob(inputs, context) + torch.log(
            self.calibration_f(context)
        )


def calibrate_likelihood_estimator(
    estimator: Flow, theta: Tensor, x: Tensor, **train_kwargs
) -> CalibratedLikelihoodEstimator:
    r"""Calibrates the likelihood estimator if partially trained on NaNs.

    It learns a calibration function $c_ζ(θ)=p(valid|θ)$ on the training data an applies it to the
    likelihood.

    Args:
        estimator: likelihood estimator with log_prob method.
        theta: parameters.
        x: observations.

    Returns:
        a calibrated likelihood function.
    """
    calibration_f = NaNCalibration(int(theta.shape[1]))
    y = includes_nan(x).float()
    calibration_f.train(theta, y, **train_kwargs)
    return CalibratedLikelihoodEstimator(deepcopy(estimator), calibration_f)
