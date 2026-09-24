use crate::{CameraIntrinsics, CameraPose};

/// Project pixel-center `(u, v)` and optical-axis depth in metres into Y-up world coordinates.
///
/// `intrinsics` must be expressed at the depth image resolution and `pose` must
/// be a column-major camera-to-world transform. Callers integrating frames
/// should use [`crate::OccupancyMapper`] to get validation and occupancy updates.
pub fn project_world(
    u: usize,
    v: usize,
    depth_m: f32,
    k: CameraIntrinsics,
    pose: CameraPose,
) -> [f32; 3] {
    let camera = [
        (u as f32 - k.cx) * depth_m / k.fx,
        -(v as f32 - k.cy) * depth_m / k.fy,
        -depth_m,
    ];
    std::array::from_fn(|r| {
        pose.columns[3][r] + (0..3).map(|c| pose.columns[c][r] * camera[c]).sum::<f32>()
    })
}
