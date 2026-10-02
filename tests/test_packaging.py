"""packaging/build_client.py: opendecider-client is the same package with no PyTorch and no local-model extras."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _builder():
    spec = importlib.util.spec_from_file_location("build_client", ROOT / "packaging" / "build_client.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_client_has_no_local_model_dependencies():
    tomllib = pytest.importorskip("tomllib")   # Python 3.11 and later
    main = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    client = tomllib.loads(_builder().client_pyproject((ROOT / "pyproject.toml").read_text()))["project"]
    assert client["name"] == "opendecider-client" and client["version"] == main["version"]
    assert client["dependencies"] == []
    extras = client["optional-dependencies"]
    assert not {"small", "mlx", "serve", "dev"} & set(extras)
    assert set(extras) == set(main["optional-dependencies"]) - {"small", "mlx", "serve", "dev"}
    assert all(extras[k] == main["optional-dependencies"][k] for k in extras)   # the rest are unchanged
    assert client["scripts"] == main["scripts"]


def test_the_client_build_stops_if_pyproject_changes_shape():
    with pytest.raises(SystemExit, match="no longer matches"):
        _builder().client_pyproject("[project]\nname = \"something-else\"\n")


def _as_client(monkeypatch, installed: bool):
    """Pretend opendecider-client is (or is not) the installed distribution."""
    import importlib.metadata as md
    real = md.distribution

    def distribution(name):
        if name == "opendecider-client" and not installed:
            raise md.PackageNotFoundError(name)
        return object() if name == "opendecider-client" else real(name)

    monkeypatch.setattr(md, "distribution", distribution)


def test_without_local_model_packages_loading_says_what_to_install(monkeypatch, tmp_path):
    """What opendecider-client users see: a local model needs the full package, said plainly, before any download."""
    import json

    import opendecider
    _as_client(monkeypatch, installed=True)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name in ("torch", "transformers", "huggingface_hub", "mlx_lm")
                        else real(name, *a))
    with pytest.raises(ImportError, match="needs the full package: pip install opendecider"):
        opendecider.load("manjunathshiva/opendecider-nano")   # a Hub name: stops before huggingface_hub is used
    folder = tmp_path / "model"
    folder.mkdir()
    for kind in ("nano", "small", "small-mlx"):   # a local folder: stops before torch / MLX is imported
        (folder / "opendecider.json").write_text(json.dumps({"kind": kind, "base_model": "x", "name": "m"}))
        with pytest.raises(ImportError, match="opendecider-client"):
            opendecider.load(str(folder))


def test_both_packages_installed_is_a_warning(monkeypatch):
    import warnings

    import opendecider
    monkeypatch.setattr(opendecider, "_installed_here", lambda: {"opendecider", "opendecider_client"})
    with pytest.warns(RuntimeWarning, match="both opendecider and opendecider-client are installed"):
        opendecider._warn_if_both_installed()
    for one in ({"opendecider"}, {"opendecider_client"}, set()):
        monkeypatch.setattr(opendecider, "_installed_here", lambda one=one: one)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            opendecider._warn_if_both_installed()   # one package (or an editable install): silent


def test_the_installed_packages_are_read_from_their_dist_info_folders(tmp_path, monkeypatch):
    import opendecider
    pkg = tmp_path / "opendecider"
    pkg.mkdir()
    (tmp_path / "opendecider-0.5.0.dist-info").mkdir()
    (tmp_path / "opendecider_client-0.5.0.dist-info").mkdir()
    (tmp_path / "opendecider_extras-1.0.dist-info").mkdir()   # another package with a similar name: not ours
    monkeypatch.setattr(opendecider, "__file__", str(pkg / "__init__.py"))
    assert {"opendecider", "opendecider_client"} <= opendecider._installed_here()
    assert "opendecider_extras" in opendecider._installed_here()   # listed, but the warning needs both exact names


def test_a_missing_package_is_logged_as_one_line(caplog):
    """A model that cannot load because a package is missing (opendecider-client asked for a local model) is a
    configuration problem: one clear log line, no traceback."""
    from opendecider.tools import Decider, ModelError

    def loader(name, **kw):
        raise ImportError("running a model on this machine needs the full package")

    with pytest.raises(ModelError, match="needs the full package"):
        Decider("manjunathshiva/opendecider-nano", loader=loader).model()
    record = next(r for r in caplog.records if "could not load" in r.getMessage())
    assert record.exc_info is None and "needs the full package" in record.getMessage()


def test_an_unreadable_site_packages_never_breaks_import(monkeypatch):
    import opendecider

    def broken():
        raise OSError("permission denied")

    monkeypatch.setattr(opendecider, "_installed_here", broken)
    opendecider._warn_if_both_installed()   # no exception


def test_the_full_package_names_the_missing_extra(monkeypatch, tmp_path):
    """In opendecider (not the client), a missing extra gets its own install line, not "install the full package"."""
    import json

    import opendecider
    _as_client(monkeypatch, installed=False)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "mlx_lm" else real(name, *a))
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "opendecider.json").write_text(json.dumps({"kind": "small-mlx", "name": "m"}))
    with pytest.raises(ImportError, match=r'this model needs mlx_lm: pip install "opendecider\[mlx\]"'):
        opendecider.load(str(folder))
