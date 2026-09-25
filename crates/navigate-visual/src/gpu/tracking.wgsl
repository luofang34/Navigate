struct Parameters {
    levels: array<vec4<u32>,4>,
    control: vec4<u32>,
}
@group(0) @binding(0) var<storage,read> reference: array<f32>;
@group(0) @binding(1) var<storage,read> query: array<f32>;
@group(0) @binding(2) var<storage,read> corners: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: Parameters;
@group(0) @binding(4) var<storage,read_write> matches: array<vec4<f32>>;

fn intensity(reverse: bool, index: u32) -> f32 {
    if reverse { return query[index]; }
    return reference[index];
}

fn sample(reverse: bool, level: u32, p: vec2<f32>) -> f32 {
    let level_info = params.levels[level];
    let xy = vec2<u32>(floor(p));
    let fraction = fract(p);
    let i = level_info.x + xy.y*level_info.y + xy.x;
    return mix(mix(intensity(reverse,i),intensity(reverse,i+1u),fraction.x),
        mix(intensity(reverse,i+level_info.y),intensity(reverse,i+level_info.y+1u),fraction.x),fraction.y);
}

fn inside(level: u32, p: vec2<f32>) -> bool {
    let size = vec2<f32>(params.levels[level].yz);
    return all(p >= vec2<f32>(6.0)) && all(p < size-vec2<f32>(6.0));
}

fn centered_patch(reverse: bool, level: u32, p: vec2<f32>) -> array<f32,81> {
    var values: array<f32,81>;
    var sum = 0.0;
    for (var i=0u; i<81u; i++) {
        let offset = vec2<f32>(f32(i%9u)-4.0,f32(i/9u)-4.0);
        values[i] = sample(reverse,level,p+offset);
        sum += values[i];
    }
    for (var i=0u; i<81u; i++) { values[i] -= sum/81.0; }
    return values;
}

fn coarse_seed(reverse: bool, level: u32, p: vec2<f32>, initial: vec2<f32>) -> vec3<f32> {
    if !inside(level,p) { return vec3<f32>(0.0); }
    var source_values = centered_patch(reverse,level,p);
    var best = vec3<f32>(0.0);
    var best_error = 1e30;
    for (var y=-8; y<=8; y++) {
        for (var x=-8; x<=8; x++) {
            let q = initial+vec2<f32>(f32(x),f32(y));
            if !inside(level,q) { continue; }
            var values = centered_patch(!reverse,level,q);
            var error = 0.0;
            for (var i=0u; i<81u; i++) { error += (source_values[i]-values[i])*(source_values[i]-values[i]); }
            if error < best_error { best_error = error; best = vec3<f32>(q,1.0); }
        }
    }
    return best;
}

fn align(reverse: bool, level: u32, p: vec2<f32>, initial: vec2<f32>) -> vec3<f32> {
    if !inside(level,p) { return vec3<f32>(0.0); }
    var source_values = centered_patch(reverse,level,p);
    var q = initial;
    for (var iteration=0u; iteration<25u; iteration++) {
        if !inside(level,q) { return vec3<f32>(0.0); }
        var values = centered_patch(!reverse,level,q);
        var h = vec3<f32>(0.0);
        var b = vec2<f32>(0.0);
        for (var i=0u; i<81u; i++) {
            let at = q+vec2<f32>(f32(i%9u)-4.0,f32(i/9u)-4.0);
            let gx = (sample(!reverse,level,at+vec2<f32>(1.0,0.0))-sample(!reverse,level,at-vec2<f32>(1.0,0.0)))*0.5;
            let gy = (sample(!reverse,level,at+vec2<f32>(0.0,1.0))-sample(!reverse,level,at-vec2<f32>(0.0,1.0)))*0.5;
            let error = source_values[i]-values[i];
            let weight = 20.0/max(abs(error),20.0);
            h += vec3<f32>(gx*gx,gx*gy,gy*gy)*weight;
            b += vec2<f32>(gx,gy)*error*weight;
        }
        let determinant = h.x*h.z-h.y*h.y;
        if determinant < 1.0 || h.x+h.z < 40.0 { return vec3<f32>(0.0); }
        let step = vec2<f32>(h.z*b.x-h.y*b.y,h.x*b.y-h.y*b.x)/determinant;
        q += step/max(length(step)/3.0,1.0);
        if length(step) < 0.02 { break; }
    }
    if !inside(level,q) { return vec3<f32>(0.0); }
    var values = centered_patch(!reverse,level,q);
    var error = 0.0;
    for (var i=0u; i<81u; i++) { error += abs(source_values[i]-values[i]); }
    return vec3<f32>(q,select(0.0,1.0,error/81.0 < 22.0));
}

fn track(reverse: bool, point: vec2<f32>, initial: vec2<f32>) -> vec3<f32> {
    var found = initial;
    var seeded = false;
    for (var index=i32(params.control.y)-1; index>=0; index--) {
        let level = u32(index);
        let scale = f32(1u<<level);
        let p = (point+vec2<f32>(0.5))/scale-vec2<f32>(0.5);
        var guess = (found+vec2<f32>(0.5))/scale-vec2<f32>(0.5);
        if !seeded {
            if !inside(level,p) { continue; }
            let seed = coarse_seed(reverse,level,p,guess);
            if seed.z == 0.0 { return vec3<f32>(0.0); }
            guess = seed.xy;
            seeded = true;
        }
        let aligned = align(reverse,level,p,guess);
        if aligned.z == 0.0 { return vec3<f32>(0.0); }
        found = (aligned.xy+vec2<f32>(0.5))*scale-vec2<f32>(0.5);
    }
    return vec3<f32>(found,select(0.0,1.0,seeded));
}

@compute @workgroup_size(64)
fn match_points(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let id = invocation.x;
    if id >= params.control.x { return; }
    matches[id] = vec4<f32>(0.0);
    let point = corners[id];
    let forward = track(false,point,point);
    if forward.z == 0.0 { return; }
    let backward = track(true,forward.xy,point);
    if backward.z == 0.0 || distance(backward.xy,point) > 1.0 { return; }
    matches[id] = vec4<f32>(forward,0.0);
}
