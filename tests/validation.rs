use rover_occupancy::{
    CameraIntrinsics, CameraPose, MapError, MapperConfig, MetricDepthFrame, OccupancyMapper,
};

fn frame<'a>(depth: &'a [f32]) -> MetricDepthFrame<'a> {
    MetricDepthFrame {
        timestamp_ns: 0,
        width: 2,
        height: 2,
        depth_m: depth,
        validity: None,
        intrinsics: CameraIntrinsics {
            fx: 1.0,
            fy: 1.0,
            cx: 0.5,
            cy: 0.5,
        },
        pose: CameraPose::identity(),
    }
}

#[test]
fn rejects_bad_depth_shape_and_focal_length() {
    let mut mapper = OccupancyMapper::new(MapperConfig::default()).unwrap();
    assert_eq!(
        mapper.integrate(&frame(&[1.0])),
        Err(MapError::DepthSizeMismatch)
    );
    let mut bad = frame(&[1.0; 4]);
    bad.intrinsics.fx = 0.0;
    assert_eq!(mapper.integrate(&bad), Err(MapError::InvalidIntrinsics));
}

#[test]
fn rejects_non_rigid_pose_and_bad_mask() {
    let mut mapper = OccupancyMapper::new(MapperConfig::default()).unwrap();
    let mut bad = frame(&[1.0; 4]);
    bad.pose.columns[0][0] = 2.0;
    assert_eq!(mapper.integrate(&bad), Err(MapError::InvalidPose));
    bad.pose = CameraPose::identity();
    bad.validity = Some(&[1]);
    assert_eq!(mapper.integrate(&bad), Err(MapError::MaskSizeMismatch));
}

#[test]
fn rejects_invalid_grid_configuration() {
    let config = MapperConfig {
        resolution_m: 0.0,
        ..MapperConfig::default()
    };
    assert!(matches!(
        OccupancyMapper::new(config),
        Err(MapError::InvalidConfiguration)
    ));
}

#[test]
fn huge_grid_is_an_error_not_an_allocation_panic() {
    let config = MapperConfig {
        width_cells: usize::MAX,
        height_cells: 1,
        ..MapperConfig::default()
    };
    assert!(matches!(
        OccupancyMapper::new(config),
        Err(MapError::InvalidConfiguration)
    ));
}
