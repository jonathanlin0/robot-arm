"""Material assets and per-rollout surface randomization for vision generation."""

from colorsys import hsv_to_rgb, rgb_to_hsv
from math import isfinite, sqrt

import mujoco
import numpy as np

from schemas.config import PaperColorVariationConfig, VisionDataGenerationConfig


PAPER_GEOM_NAMES = ("wall_x_min", "wall_x_max", "wall_y_min", "wall_y_max", "table")
SURFACE_GEOM_NAMES = ("orange_cube_geom", "blue_cube_geom", *PAPER_GEOM_NAMES)


def sample_paper_color(base_rgb: np.ndarray, settings: PaperColorVariationConfig,
                       rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float]]:
    """Vary a scene's paper color with small, centered triangular draws.

    Hue shifts are measured in turns. Saturation and brightness settings are
    maximum fractional deviations from one; returned metadata records the
    actual multiplicative scales. Warmth is the signed red/blue gain offset.
    The returned RGB is float32, matching MuJoCo's stored geom colors.
    """
    deviations = {
        name: float(rng.triangular(-maximum, 0.0, maximum)) if maximum else 0.0
        for name, maximum in settings.model_dump().items()
    }
    variations = {
        "hue_shift": deviations["hue_shift"],
        "saturation_scale": 1.0 + deviations["saturation_scale"],
        "brightness_scale": 1.0 + deviations["brightness_scale"],
        "warmth": deviations["warmth"],
    }
    base = np.asarray(base_rgb, dtype=np.float64)
    if not any(deviations.values()):
        return base.astype(np.float32), variations
    hue, saturation, brightness = rgb_to_hsv(*base)
    rgb = np.array(hsv_to_rgb(
        (hue + variations["hue_shift"]) % 1.0,
        float(np.clip(saturation * variations["saturation_scale"], 0.0, 1.0)),
        float(np.clip(brightness * variations["brightness_scale"], 0.0, 1.0)),
    ))
    rgb *= [1.0 + variations["warmth"], 1.0, 1.0 - variations["warmth"]]
    return np.clip(rgb, 0.0, 1.0).astype(np.float32), variations


def draw_clipped_val_from_gaussian(mean: float, variance: float, min_val: float,
                                   max_val: float, rng: np.random.Generator) -> float:
    """Reject Gaussian draws outside inclusive bounds; never clamp a sample.

    Despite the name, this samples a truncated Gaussian by rejection. The mean
    and variance describe the untruncated distribution; accepted samples have
    different moments. NumPy expects standard deviation, so use sqrt(variance).
    """
    if not all(isfinite(value) for value in (mean, variance, min_val, max_val)):
        raise ValueError("Gaussian parameters and bounds must be finite.")
    if variance < 0 or min_val > max_val:
        raise ValueError("Gaussian variance must be non-negative and min_val <= max_val.")
    if variance == 0:
        if not min_val <= mean <= max_val:
            raise ValueError("A zero-variance Gaussian must have its mean within the bounds.")
        return float(mean)
    if min_val == max_val:
        raise ValueError("A positive-variance Gaussian needs min_val < max_val.")
    standard_deviation = sqrt(variance)
    while True:
        value = float(rng.normal(mean, standard_deviation))
        if min_val <= value <= max_val:
            return value


def configure_vision_materials(spec: mujoco.MjSpec) -> None:
    """Give each cube, wall and table its own material without writing XML.

    Keep direct geom colors to preserve cube identities and the scene's paper
    palette. These independent materials let generation override the shared
    XML defaults separately for each wall and the tabletop.
    """
    for name in SURFACE_GEOM_NAMES:
        geom = spec.geom(name)
        material = spec.add_material(name=f"vision_{name}_material", rgba=geom.rgba.copy())
        geom.material = material.name


def randomize_materials(model: mujoco.MjModel, config: VisionDataGenerationConfig,
                        rng: np.random.Generator) -> dict[str, dict[str, float]]:
    """Sample arm and scene materials once using the configured distributions.

    Shared arm materials retain shared settings. Cubes, walls and tabletop each
    receive independent specular, shininess and reflectance samples. This does
    not modify the XML defaults used by validation.
    """
    base_id = model.body("base").id
    arm_geoms = model.body_rootid[model.geom_bodyid] == base_id
    arm_materials = sorted({int(material_id) for material_id in model.geom_matid[arm_geoms]
                            if material_id >= 0 and model.material(int(material_id)).name})
    surface_materials = {name: int(model.geom(name).matid[0]) for name in SURFACE_GEOM_NAMES}
    if any(material_id < 0 for material_id in surface_materials.values()):
        raise ValueError("Vision materials must be added with configure_vision_materials before compilation.")

    settings = {}
    for material_ids, properties in (
        (arm_materials, ("specular", "shininess")),
        (surface_materials.values(), ("specular", "shininess", "reflectance")),
    ):
        for material_id in material_ids:
            values = {}
            for name in properties:
                value = draw_clipped_val_from_gaussian(
                    mean=getattr(config, f"{name}_mean"),
                    variance=getattr(config, f"{name}_variance"),
                    min_val=getattr(config, f"{name}_min"),
                    max_val=getattr(config, f"{name}_max"),
                    rng=rng,
                )
                getattr(model, f"mat_{name}")[material_id] = value
                # Record the actual float32 value used by MuJoCo.
                values[name] = float(getattr(model, f"mat_{name}")[material_id])
            settings[model.material(material_id).name] = values
    return settings
