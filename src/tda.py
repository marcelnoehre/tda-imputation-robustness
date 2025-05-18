import ripser

def vietoris_rips_complex(data, maxdim=2):
    return ripser.ripser(data, maxdim=maxdim)
