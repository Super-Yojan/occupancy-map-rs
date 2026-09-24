"""Generate deterministic, metric MuJoCo depth frames for the Rust example."""
import json
import math
import os
import pathlib
import shutil
import sys
import tempfile


def publish_dataset(output: pathlib.Path, depth_files: list[tuple[str, bytes]], manifest: dict) -> None:
    """Write a complete dataset in a sibling staging directory, then publish it."""
    if output.exists():
        raise FileExistsError(f"dataset already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(tempfile.mkdtemp(prefix=".occupancy-stage-", dir=output.parent))
    try:
        for name, data in depth_files:
            if pathlib.Path(name).name != name:
                raise ValueError(f"depth filename must be a basename: {name}")
            (staging / name).write_bytes(data)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

def main() -> None:
    import mujoco
    import numpy as np
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 generate.py <dataset-dir>")
    output = pathlib.Path(sys.argv[1])
    scene = pathlib.Path(__file__).with_name("scene.xml")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    width, height = 160, 120
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.enable_depth_rendering()
    fovy = math.radians(60.0)
    fy = height / (2.0 * math.tan(fovy / 2.0))
    manifest = {
        "schema_version": 1,
        "width": width,
        "height": height,
        "intrinsics": {"fx": fy, "fy": fy, "cx": (width - 1) / 2, "cy": (height - 1) / 2},
        "frames": [],
    }
    depth_files = []
    for number in range(4):
        name = f"cam_{number}"
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        renderer.update_scene(data, camera=name)
        # MuJoCo's Python Renderer returns linear optical-axis depth in metres.
        depth = np.asarray(renderer.render(), dtype="<f4")
        if depth.shape != (height, width) or not np.isfinite(depth).all():
            raise RuntimeError(f"bad rendered depth for {name}")
        if number == 0 and abs(float(depth[height // 2, width // 2]) - 4.45) > 0.2:
            raise RuntimeError(f"cam_0 center wall depth does not match 4.45 m geometry")
        rotation = np.asarray(data.cam_xmat[camera_id]).reshape(3, 3)
        position = np.asarray(data.cam_xpos[camera_id])
        columns = [[float(rotation[row, col]) for row in range(3)] + [0.0] for col in range(3)]
        columns.append([float(position[0]), float(position[1]), float(position[2]), 1.0])
        filename = f"depth_{number:03}.f32le"
        depth_files.append((filename, depth.tobytes(order="C")))
        manifest["frames"].append({"timestamp_ns": number * 100_000_000, "depth_file": filename, "pose_columns": columns})
    renderer.close()
    publish_dataset(output, depth_files, manifest)


if __name__ == "__main__":
    main()
