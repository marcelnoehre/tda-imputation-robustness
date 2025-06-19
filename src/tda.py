from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceLandscape, PersistenceImage
from utils import as_batch, transform_pd
from constants import *

def vietoris_rips_complex(data):
    return VietorisRipsPersistence(
        homology_dimensions=DIMENSIONS,
        n_jobs=N_JOBS
    ).fit_transform(as_batch(data))

def persistence_landscape(pd):
    return PersistenceLandscape(
        n_jobs=N_JOBS,
    ).fit_transform(pd)[0]

def persistence_image(pd):
    return PersistenceImage(
        n_jobs=N_JOBS,
    ).fit_transform(pd)[0]

TDA = {
    VR: {FUNCTION: lambda data: vietoris_rips_complex(data)},
    PD: {FUNCTION: lambda data: transform_pd(data)},
    PL: {FUNCTION: lambda data: persistence_landscape(data)},
    PI: {FUNCTION: lambda data: persistence_image(data)}
}