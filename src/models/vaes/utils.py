import torch


def create_orthogonal_matrix(dim):
    # Generate a random matrix
    random_matrix = torch.randn(dim, dim)

    # Perform QR decomposition
    q, _ = torch.linalg.qr(random_matrix)

    return q