# MuJoCo depth fixture

The scene is a 6 m square room with a floor, four walls, and an interior pillar. Four fixed cameras at 1 m height provide a deterministic virtual trajectory. The camera uses a 60° vertical field of view and renders 160 × 120 depth frames. `generate.py` writes one little-endian `f32`-metre buffer per camera plus `manifest.json` with intrinsics and column-major camera-to-world poses.

From the repository root, after installing Python 3:

```sh
python3 -m pip install 'mujoco==3.3.7' 'numpy==2.3.3'
python3 rover-occupancy/examples/mujoco-map/generate.py /tmp/rover-occupancy-dataset
cargo run --manifest-path rover-occupancy/Cargo.toml --example mujoco_map -- /tmp/rover-occupancy-dataset /tmp/rover-occupancy-output
```

The output directory contains `occupancy.pgm` and `metadata.json`. On a headless Linux host, set `MUJOCO_GL=egl` with an available EGL driver, or `MUJOCO_GL=osmesa` if your MuJoCo installation supports OSMesa. On macOS, use the default renderer in a graphical login session.

Generation publishes a complete dataset directory only after all frames pass validation, and refuses to overwrite an existing dataset. Use a fresh output path for each run. `cam_0` checks that its center ray reaches the front wall at approximately 4.45 m; this catches a change in depth units or scene orientation.
