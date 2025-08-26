import numpy as np
from scipy.sparse.linalg import eigsh
from Functions import make_grid, second_derivative_matrix, infinite_well_potential, build_hamiltonian

if __name__ == "__main__":
    """
    This function compare between the analytical eigenvalues and numerical eigenvalues for the infinite well potential
    """
    x, dx = make_grid(-1.0, 1.0, N=800)   # box length L=2
    N = x.size
    D2 = second_derivative_matrix(N, dx)
    V = infinite_well_potential(x, L=2.0)
    H = build_hamiltonian(D2, V)
    vals, vecs = eigsh(H, k=6, which='SM')  # smallest magnitude eigenpairs
    vals_sorted = np.sort(vals)
    print("Numeric E:", vals_sorted[:4])
    # analytic for box L=2 (0 to L or -L/2 to L/2), for our symmetric box use L=2:
    L = 2.0
    E_analytic = [ (np.pi**2 * n**2) / (2 * L**2) for n in range(1,5) ]
    print("Analytic E:", E_analytic)
    error_values = []
    for i in range(len(E_analytic)):
        error = abs(vals_sorted[i] - E_analytic[i])*100/E_analytic[i]
        if error >= 1:
            error_values.append(error)

    print(f"Error%: {error_values}")
