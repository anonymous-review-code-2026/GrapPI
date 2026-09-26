from .config import GraphPIConfig, load_config

__version__ = "0.1.0"
__all__ = ["GraphPI", "GraphPIConfig", "load_config"]


def __getattr__(name):
    if name == "GraphPI":
        from .engine import GraphPI
        return GraphPI
    raise AttributeError(name)
