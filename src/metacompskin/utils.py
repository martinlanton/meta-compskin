"""Utility functions for matrix operations and tensor conversions."""

import numpy as np


def add_homogeneous_coordinate(M, dim):
    """Adds a homogeneous coordinate (1) to matrix for affine transformations.

    This utility function converts Cartesian coordinates to homogeneous
    coordinates by appending a dimension of 1s. This is required for
    applying affine transformations (rotation + translation) using
    matrix multiplication.

    Mathematical Background:
        Homogeneous coordinates allow affine transformations to be
        represented as matrix multiplication:

        [ x' ]   [ a  b  c  tx ] [ x ]
        [ y' ] = [ d  e  f  ty ] [ y ]
        [ z' ]   [ g  h  i  tz ] [ z ]
                                 [ 1 ]

        The last coordinate (1) enables translation via matrix multiplication.

    Args:
        M: Input matrix of any shape, typically vertex positions.
        dim: Axis along which to add the homogeneous coordinate.
            For vertices of shape (N, 3), use dim=1 to get (N, 4).

    Returns:
        Matrix with homogeneous coordinate added.
            Shape is M.shape with dimension `dim` increased by 1.
            The added values are all ones, dtype matches input.

    Example:
        >>> verts = np.array([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
        >>> homog = add_homogeneous_coordinate(verts, 1)
        >>> print(homog)
        [[1, 2, 3, 1],
         [4, 5, 6, 1]]  # Shape (2, 4)

    Note:
        This is used in run() to convert rest_verts (N, 3) to rest_pose (N, 4)
        for applying skinning transformations via matrix multiplication.

    References:
        - Equation 2: Linear blend skinning uses homogeneous coordinates
        - Section 2.2: Skinning formulation
    """
    x = list(M.shape)
    x[dim] = 1
    return np.concatenate([M, np.ones(x)], axis=dim).astype(M.dtype)


def npf(T):
    """Converts a PyTorch tensor to NumPy array (detached from GPU/graph).

    This utility function safely converts PyTorch tensors to NumPy arrays,
    handling GPU tensors and gradient tracking. The "npf" name stands for
    "NumPy Float" (common shorthand in the original codebase).

    Conversion Process:
        1. .detach(): Removes from computation graph (no gradients)
        2. .cpu(): Moves to CPU if on GPU
        3. .numpy(): Converts to NumPy array

    Args:
        T: PyTorch tensor to convert, can be on any device.
            May have requires_grad=True or be in computation graph.

    Returns:
        NumPy array with same data and shape as input tensor.
            Always on CPU, detached from any gradients.

    Example:
        >>> gpu_tensor = torch.randn(100, 100, device="cuda", requires_grad=True)
        >>> np_array = npf(gpu_tensor)
        >>> print(type(np_array), np_array.shape)
        <class 'numpy.ndarray'> (100, 100)

    Note:
        Used throughout the code to save results to NPZ files or compute
        error metrics. The detach() call is important to avoid keeping
        the entire computation graph in memory.

    Common Usage:
        - Saving final weights: npf(Wn).transpose()
        - Computing errors: npf(A.transpose(1, 0).reshape(...))
        - Extracting transformation matrices: npf(self.rest_pose[:, :3])
    """
    return T.detach().cpu().numpy()
