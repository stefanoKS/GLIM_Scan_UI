"""Stable reconstruction engine identifiers shared by preparation and mesh orchestration."""
NKSR = 'nksr'
VDBFUSION = 'vdbfusion'
ALGORITHMS = (VDBFUSION, NKSR)
# Jobs and API requests created before algorithm selection existed are NKSR jobs.
DEFAULT_ALGORITHM = NKSR
# Prepared-input marker that proves an engine's preparation finished.
PREPARED_MARKERS = {NKSR: 'input/nksr_input.npz', VDBFUSION: 'input/vdbfusion_prepare.json'}
# Managed process roles owned by surface reconstruction; every busy check includes
# these so one engine cannot run while unrelated capture or processing is active.
ENGINE_PROCESS_KEYS = (NKSR, 'nksr_check', VDBFUSION, 'vdbfusion_check')


def normalize_algorithm(value):
    if value is None:
        return DEFAULT_ALGORITHM
    if value not in ALGORITHMS:
        raise ValueError(f'Unknown reconstruction algorithm {value!r}; choose one of {", ".join(ALGORITHMS)}')
    return value


def process_key(algorithm):
    """Managed process role that runs the mesh worker for an engine."""
    return NKSR if normalize_algorithm(algorithm) == NKSR else VDBFUSION
