//! Live camera + YOLO26 monocular depth -> occupancy grid.
//!
//! Uses approximate intrinsics (`fx = width`, `cy`/`cx` at image center); pass
//! calibrated values for geometrically accurate output.
use image::DynamicImage;
use minifb::{Window, WindowOptions};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;
use rover_occupancy::{
    CameraIntrinsics, CameraPose, MapperConfig, MetricDepthFrame, OccupancyMapper,
};
use ultralytics_inference::visualizer::color::Colormap;
use ultralytics_inference::YOLOModel;

/// Downscale a row-major depth map to `out_w` x `out_h` with nearest sampling.
fn shrink_depth(
    depth: &ultralytics_inference::results::DepthMap,
    out_w: usize,
    out_h: usize,
) -> Vec<f32> {
    let (h, w) = depth.data.dim();
    (0..out_h)
        .map(|y| {
            let sy = y * h / out_h;
            (0..out_w)
                .map(|x| depth.data[[sy, x * w / out_w]])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect()
}

/// Draw a depth view (left) and occupancy grid (right) into one RGB buffer.
fn composite(
    depth_small: &[f32],
    out_w: usize,
    out_h: usize,
    mapper: &OccupancyMapper,
) -> Vec<u32> {
    let half = out_w / 2;
    let mut pixels = vec![0u32; out_w * out_h];
    // Left: depth, normalized over in-range values; black where invalid.
    let (lo, hi) = depth_small
        .iter()
        .copied()
        .fold((f32::MAX, f32::MIN), |(a, b), d| {
            if d > 0.0 {
                (a.min(d), b.max(d))
            } else {
                (a, b)
            }
        });
    let scale = (hi - lo).max(1e-6);
    for (i, px) in pixels.iter_mut().take(out_w * out_h).enumerate() {
        let x = i % out_w;
        let y = i / out_w;
        if x >= half {
            continue;
        }
        let d = depth_small[y * half + x];
        if d <= 0.0 {
            continue;
        }
        let [r, g, b] = Colormap::Jet.sample((d - lo) / scale);
        *px = ((r as u32) << 16) | ((g as u32) << 8) | b as u32;
    }
    // Right: occupancy grid, nearest-sampled over the grid extent.
    let grid = mapper.snapshot();
    for y in 0..out_h {
        for x in half..out_w {
            let gx = (x - half) * grid.width_cells / half;
            let gy = y * grid.height_cells / out_h;
            let color = match grid.cells[gy * grid.width_cells + gx] {
                -1 => [30, 30, 30],
                p if p >= 65 => [230, 60, 40],
                p => [((255 - p as u16 * 255 / 100) as u8), u8::MAX, u8::MAX],
            };
            pixels[y * out_w + x] =
                ((color[0] as u32) << 16) | ((color[1] as u32) << 8) | color[2] as u32;
        }
    }
    pixels
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // First camera, highest frame rate the driver offers.
    let mut camera = Camera::new(
        CameraIndex::Index(0),
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
    )?;
    camera.open_stream()?;
    let (width, height) = {
        let c = camera.camera_format();
        (c.width() as usize, c.height() as usize)
    };
    println!("camera {width}x{height}");

    // Auto-downloads yolo26n-depth.onnx on first use.
    let mut model = YOLOModel::load("yolo26n-depth.onnx")?;

    // 10 m x 10 m grid at 10 cm cells; camera sits at the world origin.
    let config = MapperConfig {
        origin_x_m: -5.0,
        origin_z_m: -5.0,
        width_cells: 100,
        height_cells: 100,
        max_depth_m: 8.0,
        ..MapperConfig::default()
    };
    let mut mapper = OccupancyMapper::new(config)?;

    // Live side-by-side view: depth (left) and occupancy grid (right).
    const OUT_W: usize = 960;
    const OUT_H: usize = 480;
    let mut window = Window::new(
        "YOLO depth | occupancy",
        OUT_W,
        OUT_H,
        WindowOptions::default(),
    )?;

    let intrinsics = CameraIntrinsics {
        // ponytail: fx=fy=width is an uncalibrated guess; pass calibrated
        // intrinsics once available or the map geometry will be distorted.
        fx: width as f32,
        fy: height as f32,
        cx: width as f32 / 2.0,
        cy: height as f32 / 2.0,
    };
    // Static camera at the world origin, looking down -Z at floor height.
    let pose = CameraPose::identity();

    let mut frame_count: u64 = 0;
    loop {
        let frame = camera.frame()?;
        let image = frame.decode_image::<RgbFormat>()?;
        let results = model.predict_image(
            &DynamicImage::ImageRgb8(image),
            format!("frame-{frame_count}"),
        )?;
        let Some(depth) = results.first().and_then(|r| r.depth.as_ref()) else {
            continue;
        };

        let (h, w) = depth.data.dim();
        let depth_flat: Vec<f32> = depth.data.iter().copied().collect();
        let stats = mapper.integrate(&MetricDepthFrame {
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos() as u64,
            width: w,
            height: h,
            depth_m: &depth_flat,
            validity: None,
            intrinsics,
            pose,
        })?;
        frame_count += 1;

        // Live side-by-side: colorized depth (left), occupancy grid (right).
        let depth_small = shrink_depth(depth, OUT_W / 2, OUT_H);
        let pixels = composite(&depth_small, OUT_W, OUT_H, &mapper);
        window.update_with_buffer(&pixels, OUT_W, OUT_H)?;
        if !window.is_open() {
            break;
        }

        println!(
            "frame {frame_count}: {w}x{h} depth, {} valid px, {} hits, {} cells changed",
            stats.valid_depth_pixels, stats.obstacle_hits, stats.changed_cells
        );

        // Print a coarse ASCII view of the grid every 30 frames (~10 s).
        if frame_count.is_multiple_of(30) {
            let grid = mapper.snapshot();
            for row in grid.cells.chunks(grid.width_cells).step_by(5) {
                let line: String = row
                    .iter()
                    .step_by(5)
                    .map(|&p: &i8| match p {
                        -1 => '.',
                        p if p >= 65 => '#',
                        _ => ' ',
                    })
                    .collect();
                println!("|{line}|");
            }
        }
        if frame_count >= 300 || !window.is_open() {
            break;
        }
    }
    Ok(())
}
