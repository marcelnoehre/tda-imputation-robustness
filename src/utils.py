import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def numeric_target_mapping(target):
    """
    Map target class labels to numeric values.

    :param list target: List of class labels.

    :return np.ndarray: Numeric mapping of class labels.
    """
    # each class maps to the index of its last occurrence (as before); the lookup is built once, not once per row
    mapping = {cls: id for id, cls in enumerate(target)}
    return np.array([mapping[cls] for cls in target])

def as_batch(data):
    return np.array(data)[None, :, :]

def transform_pd(data):
    return [[[b, d] for b, d, dim in data[0] if dim == i] for i in range(3)]

_THREAD_LIMIT = None

def _single_threaded_worker():
    """
    Limit a worker process to one thread: the OpenMP/BLAS libraries (scikit-learn's neighbour searches inside
    giotto's Vietoris-Rips, NumPy) would otherwise use every core in every worker. The libraries are loaded
    first, since a limit only applies to libraries that are already loaded.
    """
    global _THREAD_LIMIT
    import sklearn.neighbors, gtda.homology  # noqa: F401
    from threadpoolctl import threadpool_limits
    _THREAD_LIMIT = threadpool_limits(limits=1)

def process_pool(workers):
    """
    Process pool whose workers start from a fresh interpreter (spawn) and run single-threaded. Forked workers
    inherit the state of the OpenMP thread pools of the main process (scikit-learn, PyTorch) and deadlock at
    their first OpenMP call (observed: every worker stuck in scikit-learn's radius_neighbors inside giotto's
    Vietoris-Rips). The main process keeps all threads for the sequential deep imputers.
    """
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                               initializer=_single_threaded_worker)

def rates_for(missing_rates, mt):
    """Missing rates of one mechanism: missing_rates is a list shared by all mechanisms or a dict mechanism -> list."""
    return missing_rates[mt] if isinstance(missing_rates, dict) else missing_rates
