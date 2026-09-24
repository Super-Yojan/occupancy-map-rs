# rover-occupancy

[![crates.io](https://img.shields.io/crates/v/occupancy-map-rs.svg)](https://crates.io/crates/occupancy-map-rs)
[![docs.rs](https://docs.rs/rover_occupancy/badge.svg)](https://docs.rs/rover_occupancy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A small, transport-independent Rust library that accumulates a fixed 2D occupancy map from **metric optical-axis depth**, depth-aligned camera intrinsics, and a camera-to-world pose. It makes a local map; it does not estimate pose or solve SLAM.

## Installation

```sh
cargo add occupancy-map-rs
```

## Use the Rust API

```rust
use rover_occupancy::{CameraIntrinsics, CameraPose, MapperConfig, MetricDepthFrame, OccupancyMapper};

fn main() -> Result<(), rover_occupancy::MapError> {
    let mut mapper = OccupancyMapper::new(MapperConfig::default())?;
    let depth_m = [2.0_f32];
    let frame = MetricDepthFrame {
        timestamp_ns: 0,
        width: 1,
        height: 1,
        depth_m: &depth_m,
        validity: None,
        intrinsics: CameraIntrinsics { fx: 1.0, fy: 1.0, cx: 0.0, cy: 0.0 },
        pose: CameraPose::identity(),
    };
    let stats = mapper.integrate(&frame)?;
    let grid = mapper.snapshot();
    println!("{} valid pixels, {} cells", stats.valid_depth_pixels, grid.cells.len());
    Ok(())
}
```

For a real camera, supply intrinsics measured at the **depth** resolution, not the RGB resolution. The `depth_m` slice is row-major (`index = v * width + u`) and contains optical-axis Z depth in metres, not Euclidean range. Pixel coordinates refer to pixel centers, with `(0,0)` at the first pixel center. A zero byte in the optional mask invalidates a sample; zero, negative, non-finite, and configured out-of-range depth values are also skipped. A malformed frame returns `MapError` and does not change the map.

The world is Y-up; the map plane is X/Z. Camera coordinates are X-right, Y-up, and -Z-forward. `CameraPose.columns` is a 4×4 **column-major** rigid `T_world_camera`: its first three columns are the world-space camera axes, and its last column is translation in metres. The floor is the constant world-Y value `MapperConfig.floor_y_m`. Returns whose height above it is within `min_obstacle_height_m..=max_obstacle_height_m` mark occupied endpoints; other valid returns can only support free space up to the measured range. `OccupancyGrid.cells` is row-major along increasing world Z, then X: `-1` means unobserved and `0..=100` is occupancy probability. The internally accumulated log odds are clamped.

The mapper rejects grids larger than 10 million cells and angular-bin counts above 100,000 to keep allocation bounded. Ray traversal is clipped to the grid before visiting cells.

## Live camera example

The `yolo_depth` example maps from a live webcam using YOLO26 monocular depth estimation. It opens camera 0, runs the `yolo26n-depth.onnx` model (auto-downloaded on first use) on every frame, integrates the predicted metric depth into the grid, and shows a live side-by-side window: colorized depth on the left, the accumulating occupancy grid on the right (dark gray = unknown, cyan→white = free, red = occupied). Close the window to stop.

```sh
cargo run --release --example yolo_depth
```

The example uses approximate intrinsics (`fx = fy = width`, principal point at image center); pass calibrated values for geometrically accurate output. Inference speed depends on your hardware and the enabled `ultralytics-inference` acceleration features (the repository's development build enables `coreml` on Apple Silicon; CUDA, TensorRT, and others are available upstream).

## MuJoCo dataset example

The `mujoco_map` example builds a map from a deterministic recorded dataset — a 6 m room with four walls and a pillar, rendered from four fixed cameras at 1 m height (160 × 120 depth, 60° vertical FOV). See [`examples/mujoco-map/README.md`](examples/mujoco-map/README.md) to generate the dataset, then:

```sh
cargo run --example mujoco_map -- /tmp/rover-occupancy-dataset /tmp/rover-occupancy-output
```

The output directory contains `occupancy.pgm` and `metadata.json` (`unknown` = 127, `free` = 255, `occupied` = 0; rows increase with world Z, columns with world X).

## Dataset contract

`manifest.json` has `schema_version: 1`, `width`, `height`, an `intrinsics` object (`fx`, `fy`, `cx`, `cy` in depth-image pixels), and an ordered `frames` array. Each frame has `timestamp_ns`, a basename-only `depth_file`, and `pose_columns` (four arrays of four numbers). Depth files contain exactly `width * height` little-endian IEEE-754 `f32` optical-axis metre values in row-major order. Timestamps must increase. The Rust example validates paths, shape, and frame metadata before it writes the map.

## Adapting real depth sources

- **iPhone LiDAR / ARKit:** Convert each `u16` millimetre sample with `f32::from(mm) * 0.001`; mark zero/missing samples invalid. Supply the frame's camera-to-world pose and ARKit camera intrinsics. If the captured intrinsics are for a wider RGB image, scale `fx`, `fy`, `cx`, and `cy` by the corresponding depth-to-RGB width and height ratios. The current app's 256×192 depth stream needs intrinsics added by an upstream adapter; an approximate field of view is not accurate enough for mapping.
- **Metric monocular depth (including the `yolo_depth` example's YOLO26 depth model):** Convert the output tensor to optical-axis metres at its own width and height, rescale intrinsics to that resolution, and supply a synchronized VIO/SLAM pose. Check the model's claimed units and scene domain against a known distance before integrating. A depth model does **not** provide camera pose.
- **Relative monocular depth:** Raw relative values cannot be mapped as metres. An external scale/calibration stage must first produce defensible metric depth. The library intentionally does not estimate that scale.

This is a locally flat-floor map. Pose drift, dynamic objects, reflective or transparent surfaces, missing depth, wrong intrinsics, and an incorrect floor height can create false occupied or free cells. It is not yet a multi-rover fusion or navigation component.

## License

MIT — see [LICENSE](LICENSE).
