use rover_occupancy::{
    CameraIntrinsics, CameraPose, MapError, MapperConfig, MetricDepthFrame, OccupancyMapper,
};

fn config() -> MapperConfig {
    MapperConfig {
        origin_x_m: -3.0,
        origin_z_m: -3.0,
        width_cells: 12,
        height_cells: 12,
        resolution_m: 0.5,
        angular_bins: 360,
        ..MapperConfig::default()
    }
}
fn pose(x: f32) -> CameraPose {
    let mut p = CameraPose::identity();
    p.columns[3] = [x, 1.0, 0.0, 1.0];
    p
}
fn frame<'a>(depth: &'a [f32], p: CameraPose) -> MetricDepthFrame<'a> {
    MetricDepthFrame {
        timestamp_ns: 0,
        width: 1,
        height: 1,
        depth_m: depth,
        validity: None,
        intrinsics: CameraIntrinsics {
            fx: 1.0,
            fy: 1.0,
            cx: 0.0,
            cy: 0.0,
        },
        pose: p,
    }
}
fn cell(map: &rover_occupancy::OccupancyGrid, x: f32, z: f32) -> i8 {
    let col = ((x - map.origin_x_m) / map.resolution_m).floor() as usize;
    let row = ((z - map.origin_z_m) / map.resolution_m).floor() as usize;
    map.cells[row * map.width_cells + col]
}

#[test]
fn wall_hit_marks_endpoint_and_free_space() {
    let mut m = OccupancyMapper::new(config()).unwrap();
    let stats = m.integrate(&frame(&[2.0], pose(0.0))).unwrap();
    let g = m.snapshot();
    assert_eq!(stats.obstacle_hits, 1);
    assert!(cell(&g, 0.0, -2.0) > 50);
    assert!(cell(&g, 0.0, -1.0) < 50);
    assert_eq!(cell(&g, 2.0, 2.0), -1);
}

#[test]
fn invalid_depth_changes_nothing_and_bad_pose_is_atomic() {
    let mut m = OccupancyMapper::new(config()).unwrap();
    m.integrate(&frame(&[f32::NAN], pose(0.0))).unwrap();
    assert!(m.snapshot().cells.iter().all(|&c| c == -1));
    let mut bad_pose = pose(0.0);
    bad_pose.columns[0][0] = 2.0;
    let before = m.snapshot().cells;
    assert_eq!(
        m.integrate(&frame(&[2.0], bad_pose)),
        Err(MapError::InvalidPose)
    );
    assert_eq!(m.snapshot().cells, before);
}

#[test]
fn floor_return_is_not_a_hit_or_unbounded_free_ray() {
    let mut m = OccupancyMapper::new(config()).unwrap();
    let mut p = pose(0.0);
    let diagonal = std::f32::consts::FRAC_1_SQRT_2;
    p.columns[1] = [0.0, diagonal, -diagonal, 0.0];
    p.columns[2] = [0.0, diagonal, diagonal, 0.0];
    let stats = m.integrate(&frame(&[std::f32::consts::SQRT_2], p)).unwrap();
    let g = m.snapshot();
    assert_eq!(stats.considered_rays, 1);
    assert_eq!(stats.obstacle_hits, 0);
    assert!(cell(&g, 0.0, -0.5) < 50);
    assert!(g.cells.iter().all(|&c| c <= 50));
    assert_eq!(cell(&g, 0.0, -1.5), -1);
}

#[test]
fn repeated_observations_accumulate_and_motion_shifts_hits() {
    let mut m = OccupancyMapper::new(config()).unwrap();
    m.integrate(&frame(&[2.0], pose(0.0))).unwrap();
    let first = cell(&m.snapshot(), 0.0, -2.0);
    m.integrate(&frame(&[2.0], pose(0.0))).unwrap();
    assert!(cell(&m.snapshot(), 0.0, -2.0) > first);
    m.integrate(&frame(&[2.0], pose(1.0))).unwrap();
    assert!(cell(&m.snapshot(), 1.0, -2.0) > 50);
}

#[test]
fn outside_camera_clips_into_grid() {
    let mut m = OccupancyMapper::new(config()).unwrap();
    let mut outside = pose(0.0);
    outside.columns[3][2] = 4.0;
    m.integrate(&frame(&[4.0], outside)).unwrap();
    assert!(m.snapshot().cells.iter().any(|&c| c >= 0));
}

#[test]
fn nearer_hit_prevents_carving_through_wall() {
    let mut m = OccupancyMapper::new(config()).unwrap();
    let depth = [1.0, 2.0];
    let f = MetricDepthFrame {
        timestamp_ns: 0,
        width: 1,
        height: 2,
        depth_m: &depth,
        validity: None,
        intrinsics: CameraIntrinsics {
            fx: 1.0,
            fy: 1_000_000.0,
            cx: 0.0,
            cy: 0.0,
        },
        pose: pose(0.0),
    };
    m.integrate(&f).unwrap();
    let g = m.snapshot();
    assert!(cell(&g, 0.0, -1.0) > 50);
    assert_eq!(cell(&g, 0.0, -2.0), -1);
}
