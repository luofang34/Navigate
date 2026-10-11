@group(0) @binding(0) var<storage, read> p: array<f32>;
@group(0) @binding(1) var imagery: texture_2d<f32>;
@group(0) @binding(2) var terrain: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<vec2<u32>>;

// Keep offsets relative to the anchor: subtracting two f32 Earth positions
// would discard the sub-metre terrain and image precision needed by matching.
fn geographic(point: vec3<f32>) -> vec3<f32> {
    if p[35] == 0.0 { return point; }
    let r = p[34];
    let radial = r + point.z;
    let a = radial * p[33] - point.y * p[32];
    let h = sqrt(a * a + point.x * point.x);
    let dh = point.x * point.x / (h + a);
    let dlat = atan2(point.y - dh * p[32], radial + dh * p[33]);
    let dlon = atan2(point.x, a);
    let half_sin = sin(0.5 * dlat);
    let ds = sin(dlat) * p[33] - 2.0 * half_sin * half_sin * p[32];
    let t = ds / (p[33] * p[33] - p[32] * ds);
    // Metal's logarithmic atanh loses small geographic offsets to cancellation.
    // This local series has < 1.1e-11 remainder for |t| <= 0.125.
    let t2 = t * t;
    let small = t * (1.0 + t2 * (1.0 / 3.0 + t2 * (1.0 / 5.0
        + t2 * (1.0 / 7.0 + t2 / 9.0))));
    let dm = select(atanh(t), small, abs(t) <= 0.125);
    let altitude = (dot(point.xy, point.xy) + point.z * (2.0 * r + point.z))
        / (length(vec3<f32>(point.xy, radial)) + r);
    return vec3<f32>(dlon, dm, altitude);
}

fn coordinates(point: vec3<f32>, offset: u32) -> vec2<f32> {
    let scale = select(p[offset + 3u], p[offset + 2u], p[35] != 0.0);
    return vec2<f32>(p[offset], p[offset + 1u]) + vec2<f32>(point.x, -point.y) * scale;
}

fn sample_map(map: texture_2d<f32>, xy: vec2<f32>) -> vec2<f32> {
    let size = vec2<f32>(textureDimensions(map));
    if any(xy < vec2<f32>(0.0)) || any(xy > size - vec2<f32>(1.0)) {
        return vec2<f32>(0.0);
    }
    let base = vec2<i32>(floor(xy));
    let last = vec2<i32>(textureDimensions(map)) - vec2<i32>(1);
    let fraction = fract(xy);
    let a = textureLoad(map, base, 0).xy;
    let b = textureLoad(map, min(base + vec2<i32>(1, 0), last), 0).xy;
    let c = textureLoad(map, min(base + vec2<i32>(0, 1), last), 0).xy;
    let d = textureLoad(map, min(base + vec2<i32>(1, 1), last), 0).xy;
    return mix(mix(a, b, fraction.x), mix(c, d, fraction.x), fraction.y);
}

fn height_at(eye: vec3<f32>, ray: vec3<f32>, distance: f32) -> vec2<f32> {
    let point = geographic(eye + ray * distance);
    let height = sample_map(terrain, coordinates(point, 28u));
    return vec2<f32>(point.z - height.x, height.y);
}

@compute @workgroup_size(8, 8)
fn render(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= u32(p[0]) || id.y >= u32(p[1]) { return; }
    let index = id.y * u32(p[0]) + id.x;
    output[index] = vec2<u32>(0u);
    let eye = vec3<f32>(p[8], p[9], p[10]);
    let right = vec3<f32>(p[12], p[13], p[14]);
    let up = vec3<f32>(p[16], p[17], p[18]);
    let back = vec3<f32>(p[20], p[21], p[22]);
    let ray = right * ((f32(id.x) - p[4]) / p[2])
        + up * ((p[5] - f32(id.y)) / p[3]) - back;
    var above = false;
    var previous = 10.0;
    var low = 0.0;
    var high = 0.0;
    for (var step = 0u; step < u32(p[7]); step += 1u) {
        let distance = 10.0 * pow(p[6] / 10.0, f32(step) / (p[7] - 1.0));
        let height = height_at(eye, ray, distance);
        let covered = height.y > 0.9999;
        if covered && height.x <= 0.0 && above {
            low = previous;
            high = distance;
            break;
        }
        above = covered && height.x > 0.0;
        previous = distance;
    }
    if high == 0.0 { return; }
    for (var step = 0u; step < u32(p[11]); step += 1u) {
        let middle = 0.5 * (low + high);
        let height = height_at(eye, ray, middle);
        if height.y > 0.9999 && height.x <= 0.0 { high = middle; }
        else { low = middle; }
    }
    let distance = 0.5 * (low + high);
    let point = geographic(eye + ray * distance);
    let color = sample_map(imagery, coordinates(point, 24u));
    if color.y <= 0.9999 { return; }
    output[index] = vec2<u32>(u32(round(color.x * 255.0)), bitcast<u32>(distance));
}
