def copy_weights(source, target, tau):
    weights = []

    target_network_names = target.parameters().keys()
    source_network_names = source.parameters().keys()

    for target_network_name, source_network_name in zip(
        target_network_names, source_network_names
    ):

        target_network_param = target.parameters()[target_network_name]["layers"]
        source_network_param = source.parameters()[source_network_name]["layers"]

        for i, (target_network_param, source_network_param) in enumerate(
            zip(target_network_param, source_network_param)
        ):
            if not target_network_param or not source_network_param:
                continue

            target_weight = target_network_param["weight"]
            target_bias = target_network_param["bias"]
            source_weight = source_network_param["weight"]
            source_bias = source_network_param["bias"]

            weight = tau * source_weight + (1.0 - tau) * target_weight
            bias = tau * source_bias + (1.0 - tau) * target_bias

            weights.append((f"{target_network_name}.layers.{i}.weight", weight))
            weights.append((f"{target_network_name}.layers.{i}.bias", bias))

    target.load_weights(weights)
