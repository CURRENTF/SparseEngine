from typing import Any


def config_get(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def config_layer(config: Any, layer_idx: int) -> Any:
    layers = config_get(config, "per_layer_config", None)
    if not layers:
        return config
    if isinstance(layers, dict):
        index = int(layer_idx)
        return layers.get(
            index, layers.get(str(index), layers.get(f"{index:02d}", config))
        )
    return layers[layer_idx]


def config_layer_get(
    config: Any, layer_idx: int, name: str, legacy_name: str | None = None
) -> Any:
    layer = config_layer(config, layer_idx)
    missing = object()
    if layer is not config:
        value = config_get(layer, name, missing)
        if value is not missing:
            return value
    fallback_name = legacy_name or name
    if isinstance(config, dict):
        return config.get(fallback_name)
    values = vars(config)
    return (
        values[fallback_name]
        if fallback_name in values
        else getattr(config, fallback_name, None)
    )
