import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    # Convert python lists to numpy arrays
    A = np.array(a)
    B = np.array(b)

    # magnitude of A
    mag_A = np.linalg.norm(A)
    # magnitude of B
    mag_B = np.linalg.norm(B)
    
    # dot product of A and B
    dot_prd = np.dot(A, B)

    # handle 0 vectors
    if mag_A == 0 or mag_B == 0:
        return 0.0

    return float(dot_prd / (mag_A * mag_B))