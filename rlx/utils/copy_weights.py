def copy_weights(source, target, tau):
    weights = []
    for i, (target_network_param, q_network_param) in enumerate(
        zip(
            target.parameters()["layers"],
            source.parameters()["layers"],
        )
    ):
        target_weight = target_network_param["weight"]
        target_bias = target_network_param["bias"]
        q_weight = q_network_param["weight"]
        q_bias = q_network_param["bias"]

        weight = tau * q_weight + (1.0 - tau) * target_weight
        bias = tau * q_bias + (1.0 - tau) * target_bias

        weights.append((f"layers.{i}.weight", weight))
        weights.append((f"layers.{i}.bias", bias))
    target.load_weights(weights)
