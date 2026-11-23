"""H3 integration; install alongside the shared BExplains_RefPacks package."""
import importlib.util
from pathlib import Path
import sys

if "bexplains_refpacks" not in sys.modules:
    shared = Path(__file__).resolve().parent.parent / "BExplains_RefPacks" / "__init__.py"
    if not shared.is_file():
        raise ImportError("Install BExplains_RefPacks alongside BExplains_H3_RefPacks.")
    spec = importlib.util.spec_from_file_location("bexplains_refpacks", shared, submodule_search_locations=[str(shared.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
WEB_DIRECTORY = "./web"
__version__ = "1.0.0"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
