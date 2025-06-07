import pytest

from triturus.utils import ensure_reproducibility


@pytest.fixture(autouse=True)
def _setup_global_state():
    ensure_reproducibility()
