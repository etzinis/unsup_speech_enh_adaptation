""" 
Scale invariant SDR as proposed in
https://www.merl.com/publications/docs/TR2019-013.pdf
"""

import numpy as np

def compute_sisdr(estimate, reference):
    """ Compute the scale invariant SDR.

    Parameters
    ----------
    estimate : array of float, shape (n_samples,)
        Estimated signal.
    reference : array of float, shape (n_samples,)
        Ground-truth reference signal.

    Returns
    -------
    sisdr : float
        SI-SDR.
        
    Example
    --------
    >>> import numpy as np
    >>> from sisdr_metric import compute_sisdr
    >>> np.random.seed(0)
    >>> reference = np.random.randn(16000)
    >>> estimate = np.random.randn(16000)
    >>> compute_sisdr(estimate, reference)
    -48.1027283264049    
    """
    eps = np.finfo(estimate.dtype).eps
    alpha = (np.sum(estimate*reference) + eps) / (np.sum(np.abs(reference)**2) + eps)
    sisdr = 10*np.log10((np.sum(np.abs(alpha*reference)**2) + eps)/
                        (np.sum(np.abs(alpha*reference - estimate)**2) + eps))
    return sisdr