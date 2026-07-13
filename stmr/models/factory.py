"""Velocity-network registry.

Register a network builder with ``@register_model("name")`` and construct it by name
with ``get_func``. Replaces the previous if/elif dispatch.
"""

from stmr.models.siren import GroupedSiren, Siren

_MODEL_REGISTRY = {}


def register_model(name):
    def decorator(fn):
        _MODEL_REGISTRY[name] = fn
        return fn
    return decorator


@register_model("siren")
def _build_siren(**kwargs):
    # Single network shared by all frame intervals: one static velocity field.
    return Siren(**kwargs)


@register_model("groupsiren")
def _build_groupsiren(**kwargs):
    # One independent Siren per frame interval (non-stationary velocity).
    return GroupedSiren(**kwargs)


def get_func(func_name, network_kwargs):
    if func_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown velocity network {func_name!r}; "
            f"registered: {sorted(_MODEL_REGISTRY)}"
        )
    return _MODEL_REGISTRY[func_name](**network_kwargs)
