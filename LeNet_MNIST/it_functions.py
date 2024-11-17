import numpy as np

def kernel_radial_basis_function(x1, x2, h, batch_size):
    if len(x1.shape) < 1:
        d = 1
    else:
        d = x1.shape[0]
    sigma = h * np.power(batch_size, -1 / (4 + d))
    denom = (2 * sigma * sigma)
    norm_square = np.power(x1 - x2, 2)
    l2_dist = np.sum(norm_square)
    return np.exp(-l2_dist / denom)


def construct_gram_matrix(B, h: float=5):
    batch_size = B.shape[0]
    channels = B.shape[1]
    Gram = np.zeros((channels, batch_size, batch_size))
    for c in range(channels):
        for i in range(batch_size):
            xi = B[i, c]
            for j in range(i, batch_size):
                xj = B[j, c]
                kappa = kernel_radial_basis_function(xi, xj, h, batch_size)
                Gram[c, i, j] = kappa
                Gram[c, j, i] = kappa
    return Gram


def matrix_renyi_entropy(An, alpha=1.01):
    A = np.ones_like(An[0])
    for Ac in An:
        A = A * Ac
    A = A / np.trace(A)
    eigenvalues = np.linalg.eigvals(A)
    # "eigenvalues" contains very small negative values
    b1 = np.ma.masked_array(eigenvalues, eigenvalues > 0)
    b1 -= b1
    eigenvalues = np.ma.getdata(b1)
    return 1 / (1 - alpha) * np.log2(np.sum(
        np.power(eigenvalues, alpha)
    ))


def mutual_information(data, latent, alpha=1.01, h=5):
    data = data.numpy()
    latent = latent.numpy()

    G_0 = construct_gram_matrix(data, h=h)
    S_0 = matrix_renyi_entropy(G_0, alpha)

    G_n = construct_gram_matrix(latent, h=h)
    S_n = matrix_renyi_entropy(G_n, alpha)

    G_n0 = np.concatenate([G_0, G_n], axis=0)
    S_n0 = matrix_renyi_entropy(G_n0, alpha)

    return S_0 + S_n - S_n0

