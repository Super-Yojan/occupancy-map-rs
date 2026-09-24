use rover_occupancy::{CameraIntrinsics, CameraPose};

#[test]
fn optical_axis_and_corner_follow_camera_convention() {
    let k = CameraIntrinsics {
        fx: 1.0,
        fy: 1.0,
        cx: 1.0,
        cy: 1.0,
    };
    let pose = CameraPose::identity();
    let center = rover_occupancy::project_world(1, 1, 2.0, k, pose);
    let corner = rover_occupancy::project_world(2, 0, 2.0, k, pose);
    assert_eq!(center, [0.0, 0.0, -2.0]);
    assert_eq!(corner, [2.0, 2.0, -2.0]);
}

#[test]
fn column_major_pose_translation_is_applied() {
    let k = CameraIntrinsics {
        fx: 1.0,
        fy: 1.0,
        cx: 0.0,
        cy: 0.0,
    };
    let mut pose = CameraPose::identity();
    pose.columns[3] = [3.0, 4.0, 5.0, 1.0];
    assert_eq!(
        rover_occupancy::project_world(0, 0, 2.0, k, pose),
        [3.0, 4.0, 3.0]
    );
}
