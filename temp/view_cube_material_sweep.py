"""Render the project cubes with 10 specular, shininess, and reflectance values.

Run from the repository root:
    .venv/bin/python temp/view_cube_material_sweep.py
    .venv/bin/python temp/view_cube_material_sweep.py --specular-range 0 0.6

Outputs a 3-by-10 PNG grid, raw rendered tiles, and the exact preview MJCF under
temp/. The default sweeps span the full supported [0, 1] range. This is a fixed
material test scene using the project's cube dimensions/colors, with no image
augmentation, simulation randomization, or changes to the training scene.

Native MuJoCo reflects only the first reflective geom and only its local +Z
face for boxes. Both cube materials receive each value; only the orange cube
shows mirror reflections. Background stripes provide a fixed reflection target.
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
COLUMNS = 10
WIDTH, HEIGHT = 384, 320
ROW_SETTINGS = (
    ("specular", "Highlight strength", {"specular": 0.0, "shininess": 0.35, "reflectance": 0.0}),
    ("shininess", "Highlight tightness", {"specular": 0.8, "shininess": 0.0, "reflectance": 0.0}),
    ("reflectance", "Mirror reflection strength", {"specular": 0.3, "shininess": 0.35, "reflectance": 0.0}),
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    try:
        return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)
    except OSError:
        return ImageFont.load_default(size=size)


def camera_axes(position: np.ndarray, target: np.ndarray) -> str:
    backward = position - target
    backward /= np.linalg.norm(backward)
    right = np.cross([0.0, 0.0, 1.0], backward)
    right /= np.linalg.norm(right)
    upward = np.cross(backward, right)
    return " ".join(map(str, np.concatenate((right, upward))))


def make_scene() -> str:
    """Use real project cube geometry in a controlled, close-up material scene."""
    source = ET.parse(ROOT / "scenes/so101_two_cube_stack.xml").getroot()
    cubes = {}
    for color in ("orange", "blue"):
        geom = source.find(f".//geom[@name='{color}_cube_geom']")
        cubes[color] = {"size": geom.attrib["size"], "rgba": geom.attrib["rgba"]}
    position = np.array([0.09, -0.18, 0.15])
    axes = camera_axes(position, np.array([0.0, 0.0, 0.035]))
    stripes = "\n".join(
        f'<geom type="box" pos="{x} 0.119 0.13" size="0.009 0.001 0.065" '
        f'rgba="{shade} {shade} {shade} 1" material="matte"/>'
        for x, shade in ((-0.09, 0.08), (-0.045, 0.08), (0.0, 0.08), (0.045, 0.08), (0.09, 0.08))
    )
    return f'''<mujoco model="Cube material comparison">
      <visual>
        <global offwidth="{WIDTH}" offheight="{HEIGHT}"/>
        <quality shadowsize="2048" numquads="12"/>
        <headlight ambient="0.3 0.3 0.3" diffuse="0.25 0.25 0.25" specular="0 0 0"/>
        <map znear="0.005" zfar="8"/>
      </visual>
      <asset>
        <material name="orange_surface" rgba="{cubes['orange']['rgba']}"/>
        <material name="blue_surface" rgba="{cubes['blue']['rgba']}"/>
        <material name="matte" specular="0" shininess="0" reflectance="0"/>
      </asset>
      <worldbody>
        <!-- The key illuminates cube tops from behind the reflection card.
             Disable its shadows so the card does not obscure the highlights. -->
        <light name="key" pos="-0.6 1.2 0.8" dir="0.6 -1.2 -0.765"
               directional="true" diffuse="0.55 0.55 0.55" ambient="0 0 0"
               specular="0.8 0.8 0.8" castshadow="false"/>
        <camera name="comparison" pos="{' '.join(map(str, position))}"
                xyaxes="{axes}" fovy="34"/>
        <!-- Orange is first so it receives MuJoCo's one mirror-reflection pass. -->
        <geom name="orange_cube" type="box" pos="-0.034 0 0.02"
              size="{cubes['orange']['size']}" material="orange_surface"/>
        <geom name="blue_cube" type="box" pos="0.034 0 0.02"
              size="{cubes['blue']['size']}" material="blue_surface"/>
        <geom name="table" type="box" pos="0 0 -0.012" size="0.35 0.35 0.012"
              rgba="0.38 0.40 0.43 1" material="matte"/>
        <geom name="reflection_card" type="box" pos="0 0.122 0.13" size="0.16 0.002 0.08"
              rgba="0.82 0.82 0.82 1" material="matte"/>
        {stripes}
      </worldbody>
    </mujoco>'''


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "temp/cube_material_sweep.png")
    for parameter, _, _ in ROW_SETTINGS:
        parser.add_argument(f"--{parameter}-range", type=float, nargs=2, default=(0.0, 1.0),
                            metavar=("MIN", "MAX"))
    arguments = parser.parse_args()
    ranges = {name: getattr(arguments, f"{name}_range") for name, _, _ in ROW_SETTINGS}
    for name, (lower, upper) in ranges.items():
        if not 0 <= lower <= upper <= 1:
            parser.error(f"{name} range must satisfy 0 <= MIN <= MAX <= 1.")
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output.parent / f"{output.stem}_frames"
    frame_dir.mkdir(exist_ok=True)
    xml = make_scene()
    output.with_suffix(".xml").write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    material_ids = [model.material(name).id for name in ("orange_surface", "blue_surface")]

    margin, gutter, title_height, row_header, value_height, footer_height = 24, 8, 122, 62, 35, 82
    row_height = row_header + value_height + HEIGHT + 24
    grid = Image.new("RGB", (2 * margin + COLUMNS * WIDTH + (COLUMNS - 1) * gutter,
                             title_height + 3 * row_height + footer_height), "#f4f6f9")
    draw = ImageDraw.Draw(grid)
    draw.text((margin, 16), "MuJoCo cube materials | full-range comparison", font=font(44, True), fill="#111827")
    draw.text((margin, 74), "Same 4 cm project cubes. Fixed studio camera, lighting and striped reflection target. Both cubes receive each value.",
              font=font(25), fill="#475569")
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    records = []
    try:
        for row, (parameter, description, baseline) in enumerate(ROW_SETTINGS):
            y = title_height + row * row_height
            constants = "; ".join(f"{key}={value:g}" for key, value in baseline.items() if key != parameter)
            label = f"{parameter.upper()} | {description}     ({constants})"
            if parameter == "reflectance":
                label += "     Orange top only: native renderer limitation"
            draw.text((margin, y + 8), label, font=font(27, True), fill="#111827")
            for column, value in enumerate(np.linspace(*ranges[parameter], COLUMNS)):
                settings = {**baseline, parameter: float(value)}
                for name, setting in settings.items():
                    getattr(model, f"mat_{name}")[material_ids] = setting
                renderer.update_scene(data, camera="comparison")
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
                pixels = renderer.render().copy()
                frame = Image.fromarray(pixels)
                frame_path = frame_dir / f"{parameter}_{column:02d}_{value:.3f}.png"
                frame.save(frame_path)
                x = margin + column * (WIDTH + gutter)
                draw.text((x + WIDTH // 2, y + row_header), f"{value:.2f}",
                          font=font(25, True), fill="#334155", anchor="mt")
                grid.paste(frame, (x, y + row_header + value_height))
                records.append({"row": parameter, "column": column + 1, "settings": settings,
                                "image": str(frame_path)})
            print(f"Rendered {parameter}: {ranges[parameter][0]:g} to {ranges[parameter][1]:g}", flush=True)
    finally:
        renderer.close()
    footer_y = title_height + 3 * row_height
    draw.text((margin, footer_y + 4), "Reflection limitation: MuJoCo reflects only the first eligible object, and only a box's local +Z face. The blue cube is a reference.",
              font=font(25), fill="#475569")
    draw.text((margin, footer_y + 40), "Values 0 and 1 are the supported extremes. Images are direct MuJoCo renders; no exposure correction, blur or RandAugment is applied.",
              font=font(25), fill="#475569")
    grid.save(output)
    output.with_suffix(".json").write_text(json.dumps({"mujoco_version": mujoco.__version__,
        "source_scene": str(ROOT / "scenes/so101_two_cube_stack.xml"), "frames": records}, indent=2) + "\n")
    print(f"Saved 3 x {COLUMNS} grid: {output}")
    print(f"Individual renders: {frame_dir}")


if __name__ == "__main__":
    main()
