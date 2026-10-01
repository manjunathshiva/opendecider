import pytest


@pytest.fixture(autouse=True)
def _fresh_log_once():
    """Each test starts with no remembered log lines, so a message logged by an earlier test is not quietened."""
    from opendecider import tools
    tools._LOGGED.clear()
    yield
