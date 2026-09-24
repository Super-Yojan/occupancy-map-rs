use crate::MapperConfig;

pub(crate) fn cell(config: MapperConfig, x: f32, z: f32) -> Option<usize> {
    let col = ((x - config.origin_x_m) / config.resolution_m).floor() as isize;
    let row = ((z - config.origin_z_m) / config.resolution_m).floor() as isize;
    if col < 0
        || row < 0
        || col >= config.width_cells as isize
        || row >= config.height_cells as isize
    {
        None
    } else {
        Some(row as usize * config.width_cells + col as usize)
    }
}

pub(crate) fn cells(config: MapperConfig, start: [f32; 2], end: [f32; 2]) -> Vec<usize> {
    let (x0, z0, x1, z1) = match clip_segment(config, start, end) {
        Some(segment) => segment,
        None => return Vec::new(),
    };
    let resolution = f64::from(config.resolution_m);
    let origin_x = f64::from(config.origin_x_m);
    let origin_z = f64::from(config.origin_z_m);
    let mut col = ((x0 - origin_x) / resolution).floor() as isize;
    let mut row = ((z0 - origin_z) / resolution).floor() as isize;
    let end_col = ((x1 - origin_x) / resolution).floor() as isize;
    let end_row = ((z1 - origin_z) / resolution).floor() as isize;
    let dx = x1 - x0;
    let dz = z1 - z0;
    let step_x = dx.signum() as isize;
    let step_z = dz.signum() as isize;
    let next_x = origin_x + (if step_x > 0 { col + 1 } else { col }) as f64 * resolution;
    let next_z = origin_z + (if step_z > 0 { row + 1 } else { row }) as f64 * resolution;
    let mut t_max_x = if dx == 0.0 {
        f64::INFINITY
    } else {
        (next_x - x0) / dx
    };
    let mut t_max_z = if dz == 0.0 {
        f64::INFINITY
    } else {
        (next_z - z0) / dz
    };
    let t_delta_x = if dx == 0.0 {
        f64::INFINITY
    } else {
        resolution / dx.abs()
    };
    let t_delta_z = if dz == 0.0 {
        f64::INFINITY
    } else {
        resolution / dz.abs()
    };
    let mut output = Vec::new();
    for _ in 0..config
        .width_cells
        .saturating_add(config.height_cells)
        .saturating_add(2)
    {
        if col < 0
            || row < 0
            || col >= config.width_cells as isize
            || row >= config.height_cells as isize
        {
            break;
        }
        output.push(row as usize * config.width_cells + col as usize);
        if col == end_col && row == end_row {
            break;
        }
        if t_max_x < t_max_z {
            col += step_x;
            t_max_x += t_delta_x;
        } else if t_max_z < t_max_x {
            row += step_z;
            t_max_z += t_delta_z;
        } else {
            col += step_x;
            row += step_z;
            t_max_x += t_delta_x;
            t_max_z += t_delta_z;
        }
    }
    output
}

fn clip_segment(
    config: MapperConfig,
    start: [f32; 2],
    end: [f32; 2],
) -> Option<(f64, f64, f64, f64)> {
    let x0 = f64::from(start[0]);
    let z0 = f64::from(start[1]);
    let dx = f64::from(end[0]) - x0;
    let dz = f64::from(end[1]) - z0;
    let min_x = f64::from(config.origin_x_m);
    let min_z = f64::from(config.origin_z_m);
    let resolution = f64::from(config.resolution_m);
    let max_x = min_x + config.width_cells as f64 * resolution - resolution * 1e-9;
    let max_z = min_z + config.height_cells as f64 * resolution - resolution * 1e-9;
    let mut enter: f64 = 0.0;
    let mut exit: f64 = 1.0;
    for (p, q) in [
        (-dx, x0 - min_x),
        (dx, max_x - x0),
        (-dz, z0 - min_z),
        (dz, max_z - z0),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let t = q / p;
            if p < 0.0 {
                enter = enter.max(t);
            } else {
                exit = exit.min(t);
            }
            if enter > exit {
                return None;
            }
        }
    }
    Some((
        x0 + enter * dx,
        z0 + enter * dz,
        x0 + exit * dx,
        z0 + exit * dz,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grazing_ray_visits_every_crossed_cell() {
        let c = MapperConfig {
            resolution_m: 1.0,
            origin_x_m: 0.0,
            origin_z_m: 0.0,
            width_cells: 3,
            height_cells: 3,
            ..MapperConfig::default()
        };
        assert_eq!(cells(c, [0.1, 0.1], [2.1, 2.0]), vec![0, 1, 4, 5, 8]);
    }

    #[test]
    fn long_outside_segment_is_clipped_to_grid() {
        let c = MapperConfig {
            resolution_m: 1.0,
            origin_x_m: 0.0,
            origin_z_m: 0.0,
            width_cells: 3,
            height_cells: 3,
            ..MapperConfig::default()
        };
        assert_eq!(cells(c, [-1000.0, 1.5], [1000.0, 1.5]), vec![3, 4, 5]);
    }
}
