import gudhi

def vietors_rips_complex(dataset, max_dimension=3):
    rips_complex = gudhi.RipsComplex(points=dataset)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    return simplex_tree

def compute_persistence(simplex_tree):
    persistence = simplex_tree.persistence()
    return persistence
