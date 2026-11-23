def validate_config(config):
    if not isinstance(config, dict) or set(config) != {"max_states", "max_depth"}:
        raise ValueError("configuration requires exactly max_states and max_depth")
    for name, minimum, maximum in (("max_states", 1, 27), ("max_depth", 0, 7)):
        if type(config[name]) is not int or not minimum <= config[name] <= maximum:
            raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return dict(config)
