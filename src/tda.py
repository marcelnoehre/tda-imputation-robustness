from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceLandscape, PersistenceImage
from src.utils import as_batch, transform_pd
from src.constants import DIMENSIONS, N_JOBS

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

TDA_METHODS = {
    'VR': {'fn': lambda data: vietoris_rips_complex(data)},
    'PD': {'fn': lambda data: transform_pd(data)},
    'PL': {'fn': lambda data: persistence_landscape(data)},
    'PI': {'fn': lambda data: persistence_image(data)}
}