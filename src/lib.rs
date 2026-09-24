//! Convert metric depth frames and world camera poses into a local occupancy grid.
//!
//! World coordinates are Y-up. The grid lies in the X/Z plane. Camera coordinates
//! are X-right, Y-up, and -Z-forward. A frame's pose maps camera to world.
//!
//! ```
//! use rover_occupancy::{CameraIntrinsics, CameraPose, MapperConfig, MetricDepthFrame, OccupancyMapper};
//! let mut mapper = OccupancyMapper::new(MapperConfig::default())?;
//! let depth = [2.0_f32];
//! let frame = MetricDepthFrame {
//!     timestamp_ns: 0, width: 1, height: 1, depth_m: &depth, validity: None,
//!     intrinsics: CameraIntrinsics { fx: 1.0, fy: 1.0, cx: 0.0, cy: 0.0 },
//!     pose: CameraPose::identity(),
//! };
//! mapper.integrate(&frame)?;
//! assert_eq!(mapper.snapshot().width_cells, 100);
//! # Ok::<(), rover_occupancy::MapError>(())
//! ```
mod grid;
mod projection;
mod raycast;

pub use grid::{OccupancyGrid, UpdateStats};
pub use projection::project_world;

/// Pinhole intrinsics in pixels, aligned with the depth image dimensions.
#[derive(Clone, Copy, Debug)]
pub struct CameraIntrinsics {
    /// Horizontal focal length in pixels at the submitted depth resolution.
    pub fx: f32,
    /// Vertical focal length in pixels at the submitted depth resolution.
    pub fy: f32,
    /// Principal-point horizontal coordinate in pixel-center coordinates.
    pub cx: f32,
    /// Principal-point vertical coordinate in pixel-center coordinates.
    pub cy: f32,
}

/// Column-major rigid transform from camera coordinates to world coordinates.
#[derive(Clone, Copy, Debug)]
pub struct CameraPose {
    /// Four columns of `T_world_camera`; the fourth is translation in metres.
    pub columns: [[f32; 4]; 4],
}

impl CameraPose {
    /// Identity camera-to-world transform.
    pub fn identity() -> Self {
        Self {
            columns: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
}

/// Row-major optical-axis depth in metres and pose for the same instant.
pub struct MetricDepthFrame<'a> {
    /// Acquisition time in nanoseconds on the producer's clock.
    pub timestamp_ns: u64,
    /// Depth image width in pixels.
    pub width: usize,
    /// Depth image height in pixels.
    pub height: usize,
    /// Optical-axis depth in metres, row-major; invalid samples are skipped.
    pub depth_m: &'a [f32],
    /// Optional row-major mask: zero invalid, nonzero valid.
    pub validity: Option<&'a [u8]>,
    /// Intrinsics aligned to this depth image.
    pub intrinsics: CameraIntrinsics,
    /// Camera-to-world pose at the depth frame's timestamp.
    pub pose: CameraPose,
}

/// Fixed bounds and evidence settings for a Y-up world/X-Z occupancy plane.
#[derive(Clone, Copy, Debug)]
pub struct MapperConfig {
    /// Square grid-cell side length in metres.
    pub resolution_m: f32,
    /// World X coordinate of the grid's low-X boundary, in metres.
    pub origin_x_m: f32,
    /// World Z coordinate of the grid's low-Z boundary, in metres.
    pub origin_z_m: f32,
    /// Number of columns along increasing world X.
    pub width_cells: usize,
    /// Number of rows along increasing world Z.
    pub height_cells: usize,
    /// Flat floor elevation in world Y metres.
    pub floor_y_m: f32,
    /// Minimum obstacle elevation above the floor, in metres.
    pub min_obstacle_height_m: f32,
    /// Maximum obstacle elevation above the floor, in metres.
    pub max_obstacle_height_m: f32,
    /// Nearest accepted optical-axis depth, in metres.
    pub min_depth_m: f32,
    /// Farthest accepted optical-axis depth, in metres.
    pub max_depth_m: f32,
    /// Number of azimuth sectors used for nearest-hit occlusion.
    pub angular_bins: usize,
    /// Positive log-odds increment for an obstacle endpoint.
    pub hit_log_odds: f32,
    /// Negative log-odds increment for observed free space.
    pub miss_log_odds: f32,
    /// Lower accumulation clamp for log odds.
    pub min_log_odds: f32,
    /// Upper accumulation clamp for log odds.
    pub max_log_odds: f32,
}

impl Default for MapperConfig {
    fn default() -> Self {
        Self {
            resolution_m: 0.1,
            origin_x_m: -5.0,
            origin_z_m: -5.0,
            width_cells: 100,
            height_cells: 100,
            floor_y_m: 0.0,
            min_obstacle_height_m: 0.1,
            max_obstacle_height_m: 2.0,
            min_depth_m: 0.1,
            max_depth_m: 8.0,
            angular_bins: 720,
            hit_log_odds: 0.9,
            miss_log_odds: -0.4,
            min_log_odds: -4.0,
            max_log_odds: 4.0,
        }
    }
}

/// Input or configuration error. An invalid frame never changes the map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapError {
    /// Grid dimensions, bounds, height band, or evidence values are invalid.
    InvalidConfiguration,
    /// Intrinsics contain invalid focal lengths or non-finite values.
    InvalidIntrinsics,
    /// Depth length does not equal width times height.
    DepthSizeMismatch,
    /// Validity-mask length does not equal width times height.
    MaskSizeMismatch,
    /// Camera pose is non-finite or not a rigid camera-to-world transform.
    InvalidPose,
}

impl std::fmt::Display for MapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for MapError {}

/// Accumulated log-odds occupancy mapper for a fixed local map.
pub struct OccupancyMapper {
    pub(crate) config: MapperConfig,
    pub(crate) odds: Vec<f32>,
    pub(crate) observed: Vec<bool>,
}

impl OccupancyMapper {
    /// Validate the fixed metric grid and evidence configuration, then create an unknown map.
    pub fn new(config: MapperConfig) -> Result<Self, MapError> {
        validate_config(config)?;
        let count = config.width_cells * config.height_cells;
        Ok(Self {
            config,
            odds: vec![0.0; count],
            observed: vec![false; count],
        })
    }

    /// Integrate a metric frame. Invalid metadata returns [`MapError`] without map mutation.
    /// Zero, negative, non-finite, masked-out, and out-of-range depth samples add no evidence.
    pub fn integrate(&mut self, frame: &MetricDepthFrame<'_>) -> Result<UpdateStats, MapError> {
        validate_frame(frame)?;
        Ok(grid::integrate(self, frame))
    }

    /// Return an owned row-major snapshot; `-1` remains unknown.
    pub fn snapshot(&self) -> OccupancyGrid {
        grid::snapshot(self)
    }
}

fn validate_config(c: MapperConfig) -> Result<(), MapError> {
    let values = [
        c.resolution_m,
        c.origin_x_m,
        c.origin_z_m,
        c.floor_y_m,
        c.min_obstacle_height_m,
        c.max_obstacle_height_m,
        c.min_depth_m,
        c.max_depth_m,
        c.hit_log_odds,
        c.miss_log_odds,
        c.min_log_odds,
        c.max_log_odds,
    ];
    if values.iter().any(|x| !x.is_finite())
        || c.width_cells == 0
        || c.height_cells == 0
        || c.width_cells
            .checked_mul(c.height_cells)
            .is_none_or(|n| n > 10_000_000)
        || c.width_cells > isize::MAX as usize
        || c.height_cells > isize::MAX as usize
        || c.angular_bins == 0
        || c.angular_bins > 100_000
        || c.resolution_m <= 0.0
        || c.min_depth_m <= 0.0
        || c.min_depth_m >= c.max_depth_m
        || c.min_obstacle_height_m < 0.0
        || c.min_obstacle_height_m >= c.max_obstacle_height_m
        || c.hit_log_odds <= 0.0
        || c.miss_log_odds >= 0.0
        || c.min_log_odds >= c.max_log_odds
        || !(c.origin_x_m + c.width_cells as f32 * c.resolution_m).is_finite()
        || !(c.origin_z_m + c.height_cells as f32 * c.resolution_m).is_finite()
    {
        return Err(MapError::InvalidConfiguration);
    }
    Ok(())
}

fn validate_frame(f: &MetricDepthFrame<'_>) -> Result<(), MapError> {
    let n = f
        .width
        .checked_mul(f.height)
        .ok_or(MapError::DepthSizeMismatch)?;
    if f.width == 0 || f.height == 0 || f.depth_m.len() != n {
        return Err(MapError::DepthSizeMismatch);
    }
    if f.validity.is_some_and(|m| m.len() != n) {
        return Err(MapError::MaskSizeMismatch);
    }
    let k = f.intrinsics;
    if ![k.fx, k.fy, k.cx, k.cy].iter().all(|x| x.is_finite()) || k.fx <= 0.0 || k.fy <= 0.0 {
        return Err(MapError::InvalidIntrinsics);
    }
    let p = f.pose.columns;
    if p.iter().flatten().any(|x| !x.is_finite())
        || p[0][3].abs() > 1e-4
        || p[1][3].abs() > 1e-4
        || p[2][3].abs() > 1e-4
        || (p[3][3] - 1.0).abs() > 1e-4
    {
        return Err(MapError::InvalidPose);
    }
    for i in 0..3 {
        for j in 0..3 {
            let dot = (0..3).map(|r| p[i][r] * p[j][r]).sum::<f32>();
            if (dot - if i == j { 1.0 } else { 0.0 }).abs() > 1e-3 {
                return Err(MapError::InvalidPose);
            }
        }
    }
    let determinant = p[0][0] * (p[1][1] * p[2][2] - p[1][2] * p[2][1])
        - p[1][0] * (p[0][1] * p[2][2] - p[0][2] * p[2][1])
        + p[2][0] * (p[0][1] * p[1][2] - p[0][2] * p[1][1]);
    if (determinant - 1.0).abs() > 1e-3 {
        return Err(MapError::InvalidPose);
    }
    Ok(())
}
