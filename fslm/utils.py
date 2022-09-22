from typing import Tuple

import torch
from torch import Tensor


def extract_and_transform_mog(
    nn: "MDN", context: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Extracts the Mixture of Gaussians (MoG) parameters
    from the MDN at either the default x or input x.

    Extracts mixture parameters at specfic context and performs an inverse
    z-transform on them to match the input shift and mean to the MDN.

    Args:
        nn: Mixture density network with `_distribution.get_mixture_components` method.
        context: Condition for the conditional distribution of the MDN.

    Returns:
        norm_logits: Log weights for each mixture component.
        means_transformed: Means for each mixture component.
        precision_factors_transformed: Precision factors for each mixture component.
        sumlogdiag: Sum of the logs of the diagonal of the decision factors
            for each mixture component.
    """

    # extract and rescale means, mixture componenets and covariances
    dist = nn._distribution

    encoded_theta = nn._embedding_net(context)

    logits, m, prec, sumlogdiag, precfs = dist.get_mixture_components(encoded_theta)
    norm_logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    scale = nn._transform._transforms[0]._scale
    shift = nn._transform._transforms[0]._shift

    means_transformed = (m - shift) / scale

    A = scale * torch.eye(means_transformed.shape[-1])
    # precision_factors_transformed = A @ precfs
    precision_factors_transformed = precfs @ A

    logits = norm_logits
    precfs = precision_factors_transformed
    sumlogdiag = torch.sum(torch.log(torch.diagonal(precfs, dim1=2, dim2=3)), dim=2)

    return norm_logits, means_transformed, precision_factors_transformed, sumlogdiag
