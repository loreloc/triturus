import pytest

from triturus.utils import ensure_reproducibility, set_tf32_enabled


@pytest.fixture(autouse=True)
def _setup_global_state():
    set_tf32_enabled(False)
    ensure_reproducibility(determinism=True)
