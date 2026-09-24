//! Offline MuJoCo dataset reader and occupancy-map exporter.
use rover_occupancy::{
    CameraIntrinsics, CameraPose, MapperConfig, MetricDepthFrame, OccupancyMapper,
};
use serde::Deserialize;
use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
struct Manifest {
    schema_version: u32,
    width: usize,
    height: usize,
    intrinsics: Intrinsics,
    frames: Vec<Frame>,
}
#[derive(Deserialize)]
struct Intrinsics {
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
}
#[derive(Deserialize)]
struct Frame {
    timestamp_ns: u64,
    depth_file: String,
    pose_columns: [[f32; 4]; 4],
}

type Dataset = (Manifest, Vec<Vec<f32>>);

fn decode_depth(bytes: &[u8], pixels: usize) -> Result<Vec<f32>, Box<dyn Error>> {
    if bytes.len() != pixels.checked_mul(4).ok_or("depth dimensions overflow")? {
        return Err("wrong depth file length".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn load_dataset(dir: &Path) -> Result<Dataset, Box<dyn Error>> {
    let canonical_dir = fs::canonicalize(dir)?;
    let manifest: Manifest = serde_json::from_slice(&fs::read(dir.join("manifest.json"))?)?;
    if manifest.schema_version != 1 || manifest.width == 0 || manifest.height == 0 {
        return Err("unsupported or invalid manifest".into());
    }
    let pixels = manifest
        .width
        .checked_mul(manifest.height)
        .ok_or("dimensions overflow")?;
    let mut depths = Vec::with_capacity(manifest.frames.len());
    let mut previous = None;
    for frame in &manifest.frames {
        if previous.is_some_and(|p| frame.timestamp_ns <= p) {
            return Err("frame timestamps must increase".into());
        }
        previous = Some(frame.timestamp_ns);
        let path = Path::new(&frame.depth_file);
        if path.components().count() != 1 || path.file_name().is_none() {
            return Err("depth file must be a basename".into());
        }
        let resolved = fs::canonicalize(dir.join(path))?;
        if !resolved.starts_with(&canonical_dir) {
            return Err("depth file escapes dataset directory".into());
        }
        depths.push(decode_depth(&fs::read(resolved)?, pixels)?);
    }
    Ok((manifest, depths))
}

fn run(dataset: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    let (manifest, depths) = load_dataset(dataset)?;
    let config = MapperConfig {
        origin_x_m: -5.0,
        origin_z_m: -5.0,
        width_cells: 200,
        height_cells: 200,
        resolution_m: 0.05,
        floor_y_m: 0.0,
        min_obstacle_height_m: 0.1,
        max_obstacle_height_m: 2.5,
        max_depth_m: 10.0,
        ..MapperConfig::default()
    };
    let mut mapper = OccupancyMapper::new(config)?;
    for (item, depth) in manifest.frames.iter().zip(&depths) {
        let frame = MetricDepthFrame {
            timestamp_ns: item.timestamp_ns,
            width: manifest.width,
            height: manifest.height,
            depth_m: depth,
            validity: None,
            intrinsics: CameraIntrinsics {
                fx: manifest.intrinsics.fx,
                fy: manifest.intrinsics.fy,
                cx: manifest.intrinsics.cx,
                cy: manifest.intrinsics.cy,
            },
            pose: CameraPose {
                columns: item.pose_columns,
            },
        };
        mapper.integrate(&frame)?;
    }
    let grid = mapper.snapshot();
    fs::create_dir_all(output)?;
    let mut pgm = format!("P5\n{} {}\n255\n", grid.width_cells, grid.height_cells).into_bytes();
    pgm.extend(grid.cells.iter().map(|&p| {
        if p < 0 {
            127
        } else {
            255 - (p as u16 * 255 / 100) as u8
        }
    }));
    fs::write(output.join("occupancy.pgm"), pgm)?;
    let metadata = serde_json::json!({"origin_x_m":grid.origin_x_m,"origin_z_m":grid.origin_z_m,
        "resolution_m":grid.resolution_m,"width_cells":grid.width_cells,"height_cells":grid.height_cells,
        "row_order":"increasing world z", "column_order":"increasing world x",
        "encoding":{"unknown":127,"free":255,"occupied":0},"frames":manifest.frames.len()});
    fs::write(
        output.join("metadata.json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 3 {
        return Err("usage: mujoco_map <dataset-dir> <output-dir>".into());
    }
    run(&PathBuf::from(&args[1]), &PathBuf::from(&args[2]))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn little_endian_depth_rejects_wrong_length() {
        assert!(decode_depth(&[0, 0, 128, 63], 2).is_err());
        assert_eq!(decode_depth(&[0, 0, 128, 63], 1).unwrap(), vec![1.0]);
    }
    #[test]
    fn rejects_path_traversal_before_reading_depth() {
        let dir = std::env::temp_dir().join(format!("rover-occupancy-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("manifest.json"), r#"{"schema_version":1,"width":1,"height":1,"intrinsics":{"fx":1,"fy":1,"cx":0,"cy":0},"frames":[{"timestamp_ns":1,"depth_file":"../secret.bin","pose_columns":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]}"#).unwrap();
        assert!(load_dataset(&dir).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn valid_dataset_exports_map_and_metadata() {
        let dir =
            std::env::temp_dir().join(format!("rover-occupancy-valid-{}", std::process::id()));
        let out = dir.join("out");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("depth.f32le"), 2.0f32.to_le_bytes()).unwrap();
        fs::write(dir.join("manifest.json"), r#"{"schema_version":1,"width":1,"height":1,"intrinsics":{"fx":1,"fy":1,"cx":0,"cy":0},"frames":[{"timestamp_ns":1,"depth_file":"depth.f32le","pose_columns":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,1,0,1]]}]}"#).unwrap();
        run(&dir, &out).unwrap();
        let bytes = fs::read(out.join("occupancy.pgm")).unwrap();
        assert!(bytes.starts_with(b"P5\n200 200\n255\n"));
        let metadata: serde_json::Value =
            serde_json::from_slice(&fs::read(out.join("metadata.json")).unwrap()).unwrap();
        assert_eq!(metadata["frames"], 1);
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn symlink_depth_outside_dataset_is_rejected() {
        use std::os::unix::fs::symlink;
        let dir = std::env::temp_dir().join(format!("rover-occupancy-link-{}", std::process::id()));
        let external =
            std::env::temp_dir().join(format!("rover-occupancy-external-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(&external, 2.0f32.to_le_bytes()).unwrap();
        symlink(&external, dir.join("depth.f32le")).unwrap();
        fs::write(dir.join("manifest.json"), r#"{"schema_version":1,"width":1,"height":1,"intrinsics":{"fx":1,"fy":1,"cx":0,"cy":0},"frames":[{"timestamp_ns":1,"depth_file":"depth.f32le","pose_columns":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,1,0,1]]}]}"#).unwrap();
        assert!(load_dataset(&dir).is_err());
        fs::remove_dir_all(dir).unwrap();
        fs::remove_file(external).unwrap();
    }
}
