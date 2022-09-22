from __future__ import annotations

import itertools
import re
import time
from argparse import ArgumentError
from math import pi
from typing import Any, Callable, Collection, Dict, Generator, List, Optional, Tuple

import torch

from pyknos.mdn.mdn import MultivariateGaussianMDN as mdn
from sbi.inference.posteriors import MCMCPosterior, RejectionPosterior
from sbi.types import TorchTransform
from scipy import spatial

# types
from torch import Tensor

def permute_dims(
    dims_or_ndims: List[int] or int, min_ndims: int = 1, max_ndims: int = -1
) -> List[Tuple[int]]:
    """Generates all possible combinations o, 'min_len': 5, 'max_len': 15f dims given a min an max amount of dims to permute.
    Args:
        dims_or_ndims: Number of dimensions to permute.
        dims: List of dims to permute.
        min_ndims: Min number of dims per permutation, i.e. n=2 -> pairs.
        max_ndims: Max number of dims per permutation, i.e. n=2 -> pairs.

    Returns:
        combinations: List of Tuples of possible combinations.
    """
    combinations = []
    if isinstance(dims_or_ndims, list):
        xdims = dims_or_ndims
    elif isinstance(dims_or_ndims, int):
        xdims = list(range(dims_or_ndims))
    else:
        raise ArgumentError("Provide number of list of dimensions.")

    if max_ndims == -1:
        max_ndims = len(xdims)

    for L in range(min_ndims, max_ndims + 1):
        for marginals in itertools.combinations(xdims, L):
            combinations.append(list(marginals))
    return combinations


def random_subsets_of_dims(
    dims: List[int], num_subsets: int = 10, min_len: int = 1, max_len: int = 9, seed=0
) -> List:
    """Random subset of possible permutations from a superset of indexes.

    Args:
        dims: list/superset of available dimensions.
        num_subsets: Number of subsets to return.
        min_len: Min number of dims per subset.
        max_len: Max number of dims per subset.

    Returns:
        List of different legnth subsets of a superset.
    """
    if seed is not None:
        torch.manual_seed(seed)

    subsets = [torch.tensor(dims)] * num_subsets
    len_set = len(dims)
    for i, subset in enumerate(subsets):
        len_subset = int(torch.randint(min_len, max_len + 1, (1,)))
        idxs2keep = torch.multinomial(torch.ones(len_set) / len_set, len_subset)
        subsets[i] = sorted(subset[idxs2keep].tolist())
    return subsets


def skip_dims(dims: List[int], n: int = 1) -> List[List[int]]:
    """Generates list of dims where each subset of n dims is skipped once.
    Args:
        dims: List of dims to drop skip dims from.
        n: number of dims to skip at once.

    Returns:
        dim_list: List of dim subsets with skipped entries.
    """
    n_out = len(dims) - (n - 1)
    dim_list = [dims[:idx] + dims[idx + n :] for idx in range(n_out)]
    return dim_list


def condition_mog(
    condition: Tensor,
    dims: List[int],
    logits: Tensor,
    means: Tensor,
    precfs: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Finds the conditional distribution p(X|Y) for a MoG.

    Args:
        condition: An array of inputs.
        dims: Dimensions of X. The rest (Y) are set as the condition.
        logits: Log-weights of each component.
        means: Means of each component.
        precfs: Precision factors of each component.
    """
    n_mixtures, n_dims = means.shape[1:]

    mask = torch.zeros(means.shape[-1], dtype=torch.bool)
    mask[dims] = True

    y = condition[:, ~mask]
    mu_x = means[:, :, mask]
    mu_y = means[:, :, ~mask]

    precfs_xx = precfs[:, :, mask]
    precfs_xx = precfs_xx[:, :, :, mask]
    precs_xx = precfs_xx.transpose(3, 2) @ precfs_xx

    precfs_yy = precfs[:, :, ~mask]
    precfs_yy = precfs_yy[:, :, :, ~mask]
    precs_yy = precfs_yy.transpose(3, 2) @ precfs_yy

    precs = precfs.transpose(3, 2) @ precfs
    precs_xy = precs[:, :, mask]
    precs_xy = precs_xy[:, :, :, ~mask]

    means = mu_x - (
        torch.inverse(precs_xx) @ precs_xy @ (y - mu_y).view(1, n_mixtures, -1, 1)
    ).view(1, n_mixtures, -1)

    diags = torch.diagonal(precfs_yy, dim1=2, dim2=3)
    sumlogdiag_yy = torch.sum(torch.log(diags), dim=2)
    log_prob = mdn.log_prob_mog(y, torch.zeros((1, 1)), mu_y, precs_yy, sumlogdiag_yy)

    # Normalize the mixing coef: p(X|Y) = p(Y,X) / p(Y) using the marginal dist.
    new_mcs = torch.exp(logits + log_prob)
    new_mcs = new_mcs / new_mcs.sum()
    logits = torch.log(new_mcs)

    sumlogdiag = torch.sum(torch.log(torch.diagonal(precfs_xx, dim1=2, dim2=3)), dim=2)
    return logits, means, precfs_xx, sumlogdiag


# adapted from @macklab/sbi
def sample_posterior_potential(
    sample_shape: Tuple,
    potential_fn: Callable,
    theta_transform: Optional[TorchTransform] = None,
    method: str = "slice_np",
    **kwargs,
) -> Tensor:
    """Uses MCMC or rejection sampling to draw posterior samples from any
    potential that returns log_probs.

    Args:
        num_samples: Desired number of samples.
        potential_fn: Potential function used for MCMC sampling.
        theta_transform: Can be used to sample in unconstrained space.
        method: Choose between "slice_np", "slice", "slice_vectorized"
            and "rejection".
        kwargs: get passed to `MCMCPosterior.sample` or in the case of
            `method=rejection` RejectionPosterior.sample.

    Returns:
        samples: Tensor of shape (num_samples, shape_of_single_theta).
    """
    if "rej" in method.lower():
        posterior = RejectionPosterior(
            potential_fn, potential_fn.prior, theta_transform
        )
    else:
        posterior = MCMCPosterior(potential_fn, potential_fn.prior, theta_transform)

    return posterior.sample(sample_shape, **kwargs)


class KMeans:
    """Simple and minimalistic implementation of the k-means algorithm in pytorch.

    Args:
        n_clusters: How many clusters to fit centroids for.
        max_iter: Maximum number of itterations when calling `.fit`.
        random_state: initialise a specific random seed.
    Attributes:
        n_clusters: Number of clusters.
        max_iter: Maximum number of itterations when calling `.fit`.
        random_state: initialise a specific random seed.
        centroids: Stores the cluster memberships when `.fit` is called.
        labels: Stores the cluster memberships when `.fit` is called.
    """

    def __init__(
        self, n_clusters: int, max_iter: int = 100, random_state: Optional[int] = 0
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def initialise_centroids(self, X: Tensor) -> Tensor:
        """Randomly initialises cluster centroids.
        Args:
            X: Data (n_samples, num_dims).

        Returns:
            centroids: Cluster means (num_clusters, num_dims).
        """
        if self.random_state != None:
            torch.manual_seed(self.random_state)
        random_idx = torch.randperm(X.shape[0])
        centroids = X[random_idx[: self.n_clusters]]
        return centroids

    def compute_centroids(self, X: Tensor, labels: Tensor) -> Tensor:
        """Calculates cluster means based on membership of the individual data points.

        Args:
            X: Data (n_samples, num_dims).
            labels: One-hot encoded cluster membership of data points (n_samples, num_dims).

        Returns:
            Cluster means bsaed on memberships of x_i.
        """
        Ns = labels.sum(dim=0)
        return (labels / Ns).T @ X

    def compute_distance(self, X: Tensor, centroids: Tensor) -> Tensor:
        """Calculates Euclidian distances the individual data points to each cluster mean.

        Args:
            X: Data (n_samples, num_dims).

        Returns:
            Distances from x_i to mu_j (n_samples, num_clusters).
        """
        diff = X.unsqueeze(1) - centroids.repeat(X.shape[0], 1, 1)
        norm_dist = diff.norm(dim=2)
        return norm_dist.square()

    def find_closest_cluster(self, distance: Tensor) -> Tensor:
        """Calculates cluster means based on membership of the individual data points.

        Args:
            distance: Distances from x_i to mu_j (n_samples, num_clusters).

        Returns:
            labels: One-hot encoded cluster memberships of x_i.
        """
        one_hot = torch.zeros_like(distance, dtype=torch.bool)
        indices = torch.argmin(distance, dim=1).view(-1, 1)
        labels = one_hot.scatter(1, indices, 1)
        return labels

    def compute_sse(self, X: Tensor, labels: Tensor, centroids: Tensor) -> Tensor:
        """Calculates cluster sum of the squared errors.
        Args:
            X: Data (n_samples, num_dims).
            labels: One-hot encoded cluster membership of data points (n_samples, num_dims).
            centroids: Cluster means (num_clusters, num_dims).

        Returns:
            Sum of the squared errors.
        """
        n_samples = X.shape[0]
        distance = torch.zeros(n_samples)
        idxs = torch.arange(self.n_clusters).repeat(n_samples, 1)[labels == 1]
        for k in range(self.n_clusters):
            distance[idxs == k] = (X[idxs == k] - centroids[k]).norm(dim=1)
        return torch.sum(distance.square())

    def fit(self, X: Tensor):
        """Runs the kmeans algorithm on data points X.

        K-Means works be alternating between assigning data points to
        clusters based on the closest centroids and then updating the
        means of these clusters based on the new membership assigments.
        The results is a decent estimate of the data means, assuming a
        predetermined number of clusters.

        Args:
            X: Data (n_samples, num_dims).
        """
        self.centroids = self.initialise_centroids(X)
        for ittr in range(self.max_iter):
            old_centroids = self.centroids
            distances = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distances)
            self.centroids = self.compute_centroids(X, self.labels)
            if torch.all(old_centroids == self.centroids):
                break
        self.error = self.compute_sse(X, self.labels, self.centroids)


class GMM:
    """Simplistic implementation of a Gaussian Mixture Model (GMM) in pytorch.

    This class implements the functionality to fit a GMM to a set of dat X (n_samples,num_dims).
    Calling `.fit` on a set of data uses the Expectation Maximsation (EM) algorithm to
    alternate between assigning cluster responsibilities E-step and re-assesing the
    mixture parameters.

    Currently K-Means initialisation and full covariances are supported.

    Args:
        n_mixtures: How many mixture components are used in the Mixture of Gaussians.
        max_iter: Maximum number of itterations when calling `.fit`.
        random_state: initialise a specific random seed.
    Attributes:
        n_clusters: Number of clusters.
        max_iter: Maximum number of itterations when calling `.fit`.
        random_state: initialise a specific random seed.
        logits: Log weights of each component (num_mixtures).
        means: Mean vectors of each component (num_mixtures, num_dims).
        precs: Precision matrices of each component (num_mixtures, num_dims, num_dims).
    """

    def __init__(
        self, n_mixtures: Tensor, max_iter: int = 100, random_state: int = 123
    ):
        self.n_mixtures = n_mixtures
        self.max_iter = max_iter
        self.random_state = random_state

    def get_parameters(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Getter for the model parameters.
        Will only return parameters if `.fit`
        has been called before.
        """
        return self.logits, self.means, self.precs

    def precs_init(self, n_dims: int, mode: str = "unit covariances") -> Tensor:
        """Initialises the precision matrices.

        The precsion matrices can either be initialsed with unit precisions or at random.

        Args:
            n_dims: Number of dimensions of the input/precision matrix.
            mode: How to initialise the precisions. 'random" v 'unit precisions'

        Returns:
            precs: Initialised precision matrices.
        """
        precs = torch.eye(n_dims).repeat(self.n_mixtures, 1, 1)

        if "rand" in mode.lower():
            x = torch.randn((self.n_mixtures, n_dims, n_dims))
            covs = x @ x.transpose(1, 2)
            precs = torch.inverse(covs)

        return precs

    def kmeans_init(self, X: Tensor) -> Tensor:
        """Initialises the mixture means with kmeans cluster means.
        Runs the k-means algorithm on X. K-means has been implemented
        seperately in `KMeans`.

        Args:
            X: Data (n_samples,num_dims).
        Returns:
            Initialised mean of each mixture component.
        """
        kmeans = KMeans(self.n_mixtures)
        kmeans.fit(X)

        return kmeans.centroids

    def params_init(
        self,
        X: Tensor,
        n_dims: int,
        means_init: str = "kmeans",
        precs_init: str = "unit precisions",
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Initialises the all model parameters.
        All parameters are initialised before the first itteration of the EM algorithm.

        Args:
            X: Data (n_samples,num_dims).
            n_dims: Number of dimensions of the input/precision matrix.
            means_init: How to initialise the means. 'random" v 'kmeans'.
            precs_init: How to initialise the precisions. 'random" v 'unit precisions'.
        Returns:
            Initialised parameters of each mixture component.
        """
        torch.manual_seed(self.random_state)
        logits = torch.log(1 / self.n_mixtures * torch.ones(self.n_mixtures))

        means = torch.randn((self.n_mixtures, n_dims))
        if "means" in means_init.lower():
            means = self.kmeans_init(X)

        precs = self.precs_init(n_dims, precs_init)
        sumlogdiag = torch.log(torch.det(precs)) / 2

        return logits, means, precs, sumlogdiag

    def compute_lls(
        self,
        X: Tensor,
        logits: Tensor,
        means: Tensor,
        precs: Tensor,
        sumlogdiag: Tensor,
    ) -> Tensor:
        """Calculates cluster responsibilities for each x_i.

        Args:
            X: Data (n_samples, num_dims).
            logits: Log weights of each component (num_mixtures).
            means: Mean vectors of each component (num_mixtures, num_dims).
            precs: Precision matrices of each component (num_mixtures, num_dims, num_dims).
            sumlogdiag: sum of the logarithms of the diagonal precision factors.

        Returns:
            log_probs: Log likelihoods for each cluster.
            logsumexp: Normalisation constant.
        """

        n_mixtures, output_dim = means.size()
        n_samples = X.shape[0]
        X = X.view(-1, 1, output_dim)

        logits = logits.repeat(n_samples, 1)
        means = means.repeat(n_samples, 1, 1)
        precs = precs.repeat(n_samples, 1, 1, 1)
        sumlogdiag = sumlogdiag.repeat(n_samples, 1)

        # Split up evaluation into parts.
        a = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        b = -(output_dim / 2.0) * torch.log(torch.tensor(2) * pi)
        c = sumlogdiag
        d1 = (X.expand_as(means) - means).view(n_samples, n_mixtures, output_dim, 1)
        d2 = torch.matmul(precs, d1)
        d = -0.5 * torch.matmul(torch.transpose(d1, 2, 3), d2).view(
            n_samples, n_mixtures
        )
        # not stable
        log_probs = torch.log(torch.exp(a + b + c + d))
        logsumexp = torch.logsumexp(a + b + c + d, dim=-1)
        return log_probs, logsumexp

    @staticmethod
    def update_covs(X: Tensor, resp: Tensor, means: Tensor) -> Tensor:
        """Computs update to the covariance matrices in maximisation step.
        Args:
            X: Data (n_samples, num_dims).
            resp: Mixture Component responsibilities.
            means: Mixture Component means.

        Returns:
            covs: Updated covariance matrices according to mixture component
                responsibilities and means.
        """
        n_mixtures, ndims = means.shape
        covs = torch.zeros((n_mixtures, ndims, ndims))

        if X.dim() < 2:
            X = X.view(1, -1)
        X = X.T

        for k in range(n_mixtures):
            X_centered = X - means[k].view(-1, 1)
            covs[k] = (
                resp[:, k] * X_centered @ X_centered.T
            ).squeeze() + 1e-5 * torch.eye(ndims)

        return covs

    def e_step(
        self,
        X: Tensor,
        logits: Tensor,
        means: Tensor,
        precs: Tensor,
        sumlogdiag: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Performs expectation step of EM algorithm.

        In the expectation step the likelihoods of a point belonging to a
        given mixture component are calculated based on the current estimate
        of the model parameters.

        Args:
            X: Data (n_samples, num_dims).
            logits: Log weights of each component (num_mixtures).
            means: Mean vectors of each component (num_mixtures, num_dims).
            precs: Precision matrices of each component (num_mixtures, num_dims, num_dims).
            sumlogdiag: sum of the logarithms of the diagonal precision factors.

        Returns:
            log_resp: Log responsibilities for each cluster.
            sumlogresp: Sum of the log responsibilities.
        """
        lls, logsum = self.compute_lls(X, logits, means, precs, sumlogdiag)

        log_resp = (lls.T - logsum).T
        sumlogresp = torch.sum(log_resp)

        return log_resp, sumlogresp

    def m_step(self, X: Tensor, log_resp: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Performs maximisation step of EM algorithm.

        In the maximisation the new model parameters are calculated based on the
        MAP estimate given the Data and responsibilities.

        Args:
            X: Data (n_samples, num_dims).
            log_resp: Log responsibilities.

        Returns:
            logits: Log weights of each component (num_mixtures).
            means: Mean vectors of each component (num_mixtures, num_dims).
            precs: Precision matrices of each component (num_mixtures, num_dims, num_dims).
            sumlogdiag: sum of the logarithms of the diagonal precision factors.
        """
        n_samples, ndims = X.shape
        resp = torch.exp(log_resp)
        sum_resp = resp.sum(0)

        logits = torch.log(sum_resp) - torch.log(torch.tensor(n_samples).float())

        means = (resp.T @ X) / sum_resp.view(-1, 1)

        covs = self.update_covs(X, resp, means) / sum_resp.view(-1, 1, 1)
        precs = torch.inverse(covs)  # possibly needs to be stabalised
        sumlogdiag = torch.log(torch.det(precs)) / 2

        return logits, means, precs, sumlogdiag

    def fit(
        self,
        X: Tensor,
        means_init: str = "kmeans",
        precs_init: str = "unit precisions",
        verbose: bool = False,
    ):
        """Fits GMM to data X..

        Runs the EM algorithm unitl convergence or itteration limit.

        Args:
            X: Data (n_samples, num_dims).
            means_init: How to initialise the means. 'random" v 'kmeans'
            precs_init: How to initialise the precisions. 'random" v 'unit precisions'
            verbose: Whether to print results of fitting process.
        """
        n_samples, ndims = X.shape

        # init
        self.logits, self.means, self.precs, sumlogdiag = self.params_init(
            X, ndims, means_init, precs_init
        )

        # EM
        sumlogresp_prev = 0
        for i in range(self.max_iter):
            log_resp, sumlogresp = self.e_step(
                X, self.logits, self.means, self.precs, sumlogdiag
            )
            self.logits, self.means, self.precs, sumlogdiag = self.m_step(X, log_resp)
            ll_change = abs(sumlogresp_prev - sumlogresp)
            sumlogresp_prev = sumlogresp
            if verbose:
                print(
                    "iter: {0} --- likelihoods: {1:.2f} --- change: {2:.2f}".format(
                        i, sumlogresp, ll_change
                    )
                )
            if ll_change < 1e-4:
                break

    def bic_or_bic(self):
        """PLACEHOLDER FOR INFORMATION CRITERION"""
        pass


# NOT TESTED OR VERIFIED
def combine_MoG(logits: Tensor, means: Tensor, precfs: Tensor) -> Tuple[Tensor, Tensor]:
    """Combine MoG into single Gaussian distribution.

    Averages over means and precisions of Mixture of Gaussians and
    returns single mean and precision.

    Args:
        logits: Log wheights of MoG (batchsize, num_mixtures).
        means: Means of MoG (batchsize, num_mixtures, num_dims)
        precfs: Precison factors of MoG (batchsize, num_mixtures, num_dims, num_dims).
    Returns:
        mu: New mean vector.
        prec: New precidion matrix.
    """
    S = precfs.transpose(2, 3) @ precfs
    mu = torch.sum(torch.exp(logits).unsqueeze(2) * means, dim=1)
    Sigma = torch.sum(
        torch.einsum("ij, klmn -> ijmn", torch.exp(logits), S)
        + torch.einsum("ijk, lmn -> imkn", (means - mu), (means - mu)),
        dim=[0, 1],
    )
    return mu, torch.inverse(Sigma)


def nearest_neighbours(X: Tensor, Y: Tensor) -> Tuple[Tensor, Tensor]:
    """Computes all nearest neighbour for all x_i in X given Y.

    Uses k-d trees to find nearest neighbours.

    Args:
        X: Sample for which to compute closest y_i for each x_i.
        Y: Sample used to look for nearest neighbours.

    Returns:
        Nearest neighbours in Y.
        Indices of nearest neighbours in Y.
    """
    tree = spatial.KDTree(Y)

    k = 2
    if torch.all(X != Y):
        k = 1

    def nn_Y(x):
        d, idx = tree.query(x, k=k, p=2)
        idx = torch.tensor(idx).view(-1, 1)
        d = torch.tensor(d).view(-1, 1)
        return torch.hstack([idx[k - 1], d[k - 1]])

    nns = [nn_Y(x) for x in X]
    nns = torch.vstack(nns)
    nn_idxs = list(nns[:, 0].int())
    return Y[nn_idxs], nns[:, 1]


def extract_tags(dct: Dict) -> List:
    """Return unique dict labels of labels with the form 'some_text_[numbers]'.

    Args:
        dct: Dictionary with labels of the form: some_text_[numbers]'

    Returns:
        unique labels.
    """
    keys = list(dct.keys())

    unique = lambda l: list(set(l))

    tags = [key[: key.find("[")] for key in keys]

    return unique(tags)


def select_tag(dct: Dict, tag: str) -> Dict:
    """Selects all entries of a dictionary matching a certain tag.
    Args:
        dct: Dictionary with labels of the form: some_text_[numbers]'
        tag: Tag, i.e. 'some_text' that is contained in the dict keys.

    Returns:
        dict with only the matching labels.
    """

    # prevents keys from being chosen, which contain additional text b4 the tag
    def condition(tag, key):
        match = re.search(tag, key)
        cond_1 = match != None
        # cond_2 = False
        # if cond_1:
        #     cond_2 = match.span()[0] == 0
        return cond_1  # and cond_2

    return dict((key, dct[key].clone()) for key in dct.keys() if condition(tag, key))


def sort_by_missing_dims(sample_dct: Dict, dim_range: "Range") -> Tuple[Tensor, Tensor]:
    """Sorts dictionary by the dims that are missing in its keys.

    Args:
        sample_dct: Contains dict of samples and labels that include feature sets.
        dim_range: Range for the total number of features used.

    Returns:
        labels: Labels the missing dims of the corresponing data in data.
        data: contains the data.
    """
    relabeled_dct = {}
    for key, value in sorted(sample_dct.items()):
        match = re.search("\[[^\]]*\]", key)
        # match = re.search("^\(([0-9])\d*\)", key)
        # match = re.search("\[([0-9]+(,[0-9]+)+)\]*", key)
        num_out = skipped_num(key[match.span()[-2] : match.span()[-1]], dim_range)[0]
        relabeled_dct["_" + str(num_out)] = value
    labels = []
    data = []

    for key, value in sorted(relabeled_dct.items()):
        match = re.search("_-*[0-9]\d*", key)
        labels.append(torch.tensor(int(match.group()[1:])))
        data.append(value)

    labels = torch.hstack(labels)
    data = torch.stack(data)

    return labels, data


def ints_from_str(string: str) -> List[int]:
    """Extracts list of integers present in string."""
    number_strs = re.findall(r"\d+", string)
    return [int(s) for s in number_strs]


def skipped_num(label: str, num_range: "Range") -> List[int]:
    """Finds all numbers missing in a label that contains a range of numbers.
    1st all numbers in a label string are extracted. Then it is compared
    to the expected range of numbers that should be present.
    Those that are not are returned.

    Args:
        label: string containing list of numbers.
        num_range: Range of numbers supposed to be in a string.

    Returns:
        missing_num: List of numbers missing from label.
    """
    numbers = ints_from_str(label)
    missing_num = []
    for i in num_range:
        if i not in numbers:
            missing_num.append(i)
    if missing_num == []:
        missing_num.append(-1)
    return missing_num


def expand_equally(*Xs: Collection[Tensor], dim: int = 0) -> Collection[Tensor]:
    """Takes a number of tensors as arguments and expands smaller ones to fit
    larger ones for a specified number of dims.
    Args:
        Xs: A Collection of Tensors, which to expand to match the tensor with the
            largest number of dimensions.
        dim: Which dimension of the tensor to expand equally.
    """
    max_dim = max([x.dim() for x in Xs])
    shapes = torch.zeros((len(Xs), max_dim))

    for i, x in enumerate(Xs):
        x_shape = torch.tensor(x.shape)
        shapes[i, 0 : len(x_shape)] = x_shape

    sizes2expand = shapes[:, dim]
    expand2 = int(max(sizes2expand))

    expandable = len(shapes[:, dim].unique()) == 2 and 1 in shapes[:, dim].unique()
    equal_sizes = len(shapes[:, dim].unique()) == 1
    assert (
        expandable or equal_sizes
    ), "more than 1 tensor has a size different from 1 on axis {}.".format(dim)

    expanded_Xs = []

    if not equal_sizes:
        for x in Xs:
            if x.shape[dim] < expand2:
                new_size = [1] * x.dim()
                new_size[dim] = expand2
                x = x.repeat(new_size)
                expanded_Xs.append(x)
            else:
                expanded_Xs.append(x)
    else:
        expanded_Xs = Xs

    return expanded_Xs


def feature_stats(
    tagged_samples: Dict, tag: Optional[str] = None, subset_size: Optional[int] = None
) -> Tuple[Tensor, Tensor]:
    """Occurance of a feature and how many times it occurs in a dataset of subsets.
    Helpful, when trying to get an overview of the distribution of features in a sample.

    Args:
    tagged_samples: Dictionary of posterior samples with tags containing:
        1. Idxs of features used, i.e. [12,15,18]
        2. Idxs of features that are part of the posterior estimate, i.e. [0,2]
        The tags can also contain some other descriptive string.
    tag: Picks up on other descriptive tags and only selects those, i.e. direct vs posthoc.
    subset_size: Only use features subsets of a fixed size, if the sample contains
        subsets of different sizes as well. If not selected, all subset sizes
        will be selected.

    Returns:
        all_features: Features that are present.
        feature_counts: Number of occurances in dataset.
    """
    feature_sets = []
    num_params = 0
    if tag != None:
        tagged_samples = select_tag(tagged_samples, tag)
    for key in tagged_samples.keys():
        match = re.findall("(\[.+\]).*(\[.+\])$", key)
        matching_tags = re.search("(\[.+\]).*(\[.+\])$", key)
        if num_params == 0 and matching_tags != None:
            num_params = tagged_samples[matching_tags.string].shape[1]
        if num_params != 0 and matching_tags != None:
            assert tagged_samples[matching_tags.string].shape[1] == num_params, (
                "Not all samples have param dim %s." % num_params
            )

        if match != []:
            feature_sets.append(match[0][0])

    feature_sets = list(set(feature_sets))
    feature_sets = [ints_from_str(subset) for subset in feature_sets]
    if subset_size != None:
        feature_sets = [ft_set for ft_set in feature_sets if len(ft_set) == subset_size]
    cum_features = [y for x in feature_sets for y in x]
    all_features = list(set(cum_features))
    feature_counts = [cum_features.count(i) for i in all_features]

    return all_features, feature_counts
    

class Timer:
    r"""Provides Stopwatch functionality to measure execution time of subroutines.

    Example:
    ```
    stopwatch = Timer()

    stopwatch.start()
    subroutine()
    stopwatch.stop()

    print(stopwatch.T)
    ```

    Provides additional functionalities to tag the stopped times and save them
    to a file.

    Attributes:
        log: Records stopped times.
        hist: Saved times that have been tagged are recorded here.
        t_start: `time.perf_counter()` at start.
        t_end: `time.perf_counter()` at the end.
        T: Time intervall.
        running: Whether the timer is currently running.
    """

    def __init__(self):
        self.log = []
        self.hist = {}
        self.t_start = None
        self.t_end = None
        self.T = None
        self.running = False

    def start(self):
        """Start the timer."""
        assert ~self.running, "The timer is already running."
        self.running = True
        self.t_start = time.perf_counter()

    def stop(self, reset: bool = True) -> float:
        """Stop the timer.

        Args:
            reset: Whether to reset the timer after stopping it.

        Returns:
            T: Time taken between start and stop in seconds.
        """
        self.t_end = time.perf_counter()
        self.T = self.t_end - self.t_start
        self.log.append(self.T)
        T = self.T
        if reset:
            self.reset()
        self.running = False
        return T

    def show(self) -> float:
        """Display elapsed time without stopping timer.

        Returns:
            T: Elapsed time.
        """
        if self.running:
            return time.perf_counter() - self.t_start
        else:
            return self.T

    def reset(self):
        """Reset the timer."""
        self.t_start = None
        self.t_end = None
        self.T = None
        self.running = False

    def stop_and_tag(self, tag: str, reset: bool = True):
        """Stops the timer and

        Args:
            tag: descriptive tag for the stopped time.
            reset: Whether to reset the timer after stopping it.
        """
        self.stop(reset)
        self.tag(tag)

    def tag(self, tag):
        """Add desriptive tag to the last acquired time interval.

        Args:
            tag: descriptive tag for the stopped time.
        """
        self.hist[tag] = self.log[-1]

    def save_log(self, logfile: str):
        """Save log to file.

        Args:
            logfile: Path to file
        """
        with open(logfile, "a") as f:
            for T in self.log:
                f.write(f"{T}\n")

    def save_hist(self, histfile: str):
        """Save tagged time intervalls to file.

        Args:
            histfile: Path to file
        """
        with open(histfile, "a") as f:
            for tag, T in self.hist.items():
                f.write(f"{tag}: {T}\n")

    def import_hist(self, histfile: str) -> dict:
        """Import tagged time intervalls from file.

        Args:
            histfile: Path to file
        """
        dct = {}
        with open(histfile, "a") as f:
            lines = f.readlines()
        for line in lines:
            key, value = line.split(": ")
            dct[key] = float(value)
        return dct


def record_scaler(x: float, tag: str, file: str):
    """Store scalar value in file and add descriptive tag.

    Args:
        x: scalar
        tag: descriptive tag for the scalar
        file: path to file
    """
    with open(file, "a") as f:
        f.write(f"{tag}: {x} \n")


def now() -> float:
    r"""Return time

    Returns:
        `time.perf_counter()`
    """
    return time.perf_counter()


def ensure_same_batchsize(*args) -> Generator:
    """Ensures same batchsize for Tensors or Lists

    If several Tensors are passed as arguments, the 0th dimension is repeated
    until it matches that of the Tensor with the largest batchsize.

    If the arguments are lists, the same lists are repeated until they have the
    length.

    Arguments should be of same type and idally the larger batch should be some multiple of
    the smaller batches.

    Yields:
        Generator of of input args rescaled to same batchsize.
    """
    same_input_type = all([isinstance(args[0], type(arg)) for arg in args[1:]])
    assert same_input_type, "Input types are not the same"
    if isinstance(args[0], Tensor):
        # for arg in args:
        #     if arg.dim() < 2:
        #         arg = arg.unsqueeze(0)
        batchsize = max([arg.shape[0] for arg in args])
        new_shapes = [torch.Size([batchsize]) + arg.shape[1:] for arg in args]
        return (arg.expand(new_shape) for arg, new_shape in zip(args, new_shapes))

    elif isinstance(args[0], List):
        batchsize = max([len(arg) for arg in args])
        return (
            arg * (int(batchsize / len(arg))) + arg[: batchsize % len(arg)]
            for arg in args
        )


