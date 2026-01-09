"""GUI settings -> Pose2Sim config dict (Config.toml-free)."""

def build_pose2sim_config(settings: dict) -> dict:
    # This is a thin shim. It keeps existing keys as-is, and sets fixed pose_model.
    cfg = dict(settings or {})
    cfg.setdefault("pose", {})
    # pose_model fixed to HALPE_26 / Body_with_feet
    cfg["pose"]["pose_model"] = "Body_with_feet"
    return cfg
