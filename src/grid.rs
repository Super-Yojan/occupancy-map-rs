use crate::{projection::project_world, raycast, MapperConfig, MetricDepthFrame, OccupancyMapper};

/// Owned row-major map in the X/Z plane. `-1` means unknown; `0..=100` is occupancy probability.
pub struct OccupancyGrid {
    /// Low-X world boundary in metres.
    pub origin_x_m: f32,
    /// Low-Z world boundary in metres.
    pub origin_z_m: f32,
    /// Cell side length in metres.
    pub resolution_m: f32,
    /// Number of columns along increasing world X.
    pub width_cells: usize,
    /// Number of rows along increasing world Z.
    pub height_cells: usize,
    /// Row-major probabilities: `-1` unknown, `0..=100` observed occupancy.
    pub cells: Vec<i8>,
}

/// Counts from one integrated depth frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UpdateStats {
    /// In-range finite depth samples with a valid mask entry.
    pub valid_depth_pixels: usize,
    /// Angular sectors with usable projected returns.
    pub considered_rays: usize,
    /// Occupied endpoints that fell inside the grid.
    pub obstacle_hits: usize,
    /// Cells newly observed or whose internal log odds changed.
    pub changed_cells: usize,
}

#[derive(Clone, Copy)]
struct Return {
    x: f32,
    z: f32,
    distance: f32,
    hit: bool,
}

pub(crate) fn integrate(mapper: &mut OccupancyMapper, frame: &MetricDepthFrame<'_>) -> UpdateStats {
    let config = mapper.config;
    let start = [frame.pose.columns[3][0], frame.pose.columns[3][2]];
    let mut bins: Vec<Vec<Return>> = vec![Vec::new(); config.angular_bins];
    let mut stats = UpdateStats::default();
    for (index, &depth) in frame.depth_m.iter().enumerate() {
        if !depth.is_finite()
            || depth < config.min_depth_m
            || depth > config.max_depth_m
            || frame.validity.is_some_and(|m| m[index] == 0)
        {
            continue;
        }
        stats.valid_depth_pixels += 1;
        let point = project_world(
            index % frame.width,
            index / frame.width,
            depth,
            frame.intrinsics,
            frame.pose,
        );
        if point.iter().any(|x| !x.is_finite()) {
            continue;
        }
        let dx = point[0] - start[0];
        let dz = point[2] - start[1];
        let distance = dx.hypot(dz);
        if distance <= 1e-6 || !distance.is_finite() {
            continue;
        }
        let angle = dz.atan2(dx).rem_euclid(std::f32::consts::TAU);
        let bin = ((angle / std::f32::consts::TAU * config.angular_bins as f32).floor() as usize)
            .min(config.angular_bins - 1);
        let height = point[1] - config.floor_y_m;
        bins[bin].push(Return {
            x: point[0],
            z: point[2],
            distance,
            hit: height >= config.min_obstacle_height_m && height <= config.max_obstacle_height_m,
        });
    }
    let mut misses = vec![false; mapper.odds.len()];
    let mut hits = vec![false; mapper.odds.len()];
    for observations in bins.iter_mut().filter(|b| !b.is_empty()) {
        stats.considered_rays += 1;
        observations.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        let obstacle = observations.iter().find(|r| r.hit).copied();
        if let Some(hit) = obstacle {
            if let Some(index) = raycast::cell(config, hit.x, hit.z) {
                hits[index] = true;
                stats.obstacle_hits += 1;
            }
        }
        let free_end = obstacle.unwrap_or(*observations.last().unwrap());
        let path = raycast::cells(config, start, [free_end.x, free_end.z]);
        for index in path {
            if obstacle.is_none_or(|o| raycast::cell(config, o.x, o.z) != Some(index)) {
                misses[index] = true;
            }
        }
    }
    for index in 0..mapper.odds.len() {
        if misses[index] || hits[index] {
            let before = mapper.odds[index];
            let delta = if hits[index] {
                config.hit_log_odds
            } else {
                config.miss_log_odds
            };
            mapper.odds[index] = (before + delta).clamp(config.min_log_odds, config.max_log_odds);
            if !mapper.observed[index] || before != mapper.odds[index] {
                stats.changed_cells += 1;
            }
            mapper.observed[index] = true;
        }
    }
    stats
}

pub(crate) fn snapshot(mapper: &OccupancyMapper) -> OccupancyGrid {
    let c: MapperConfig = mapper.config;
    let cells = mapper
        .odds
        .iter()
        .zip(&mapper.observed)
        .map(|(&odds, &observed)| {
            if !observed {
                -1
            } else {
                (100.0 / (1.0 + (-odds).exp())).round().clamp(0.0, 100.0) as i8
            }
        })
        .collect();
    OccupancyGrid {
        origin_x_m: c.origin_x_m,
        origin_z_m: c.origin_z_m,
        resolution_m: c.resolution_m,
        width_cells: c.width_cells,
        height_cells: c.height_cells,
        cells,
    }
}
