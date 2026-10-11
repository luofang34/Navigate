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


var<workgroup> source_patch: array<f32,81>;
var<workgroup> candidates: array<vec2<f32>,64>;
var<workgroup> search_point: vec2<f32>;
var<workgroup> search_initial: vec2<f32>;
var<workgroup> search_level: u32;
var<workgroup> search_active: u32;
var<workgroup> forward_result: vec3<f32>;

// Select the coarsest level with a complete source patch.
fn prepare_search(reverse: bool, point: vec2<f32>) {
    search_active = 0u;
    search_level = 0u;
    search_point = vec2<f32>(0.0);
    search_initial = vec2<f32>(0.0);
    if reverse && forward_result.z == 0.0 { return; }
    var origin = point;
    if reverse { origin = forward_result.xy; }
    for (var index=i32(params.control.y)-1; index>=0; index--) {
        let level = u32(index);
        let scale = f32(1u<<level);
        let p = (origin+vec2<f32>(0.5))/scale-vec2<f32>(0.5);
        if inside(level,p) {
            search_point = p;
            search_initial = (point+vec2<f32>(0.5))/scale-vec2<f32>(0.5);
            search_level = level;
            search_active = 1u;
            break;
        }
    }
}

fn search_candidates(reverse: bool, lane: u32) {
    if search_active == 1u {
        for (var i=lane; i<81u; i+=64u) {
            let offset = vec2<f32>(f32(i%9u)-4.0,f32(i/9u)-4.0);
            source_patch[i] = sample(reverse,search_level,search_point+offset);
        }
    }
    workgroupBarrier();
    var best_error = 1e30;
    var best_index = 289u;
    if search_active == 1u {
        for (var candidate=lane; candidate<289u; candidate+=64u) {
            let q = search_initial+vec2<f32>(f32(candidate%17u)-8.0,f32(candidate/17u)-8.0);
            if !inside(search_level,q) { continue; }
            var sum = 0.0;
            var squared = 0.0;
            for (var i=0u; i<81u; i++) {
                let offset = vec2<f32>(f32(i%9u)-4.0,f32(i/9u)-4.0);
                let difference = source_patch[i]-sample(!reverse,search_level,q+offset);
                sum += difference;
                squared += difference*difference;
            }
            let error = squared-sum*sum/81.0;
            if error < best_error { best_error = error; best_index = candidate; }
        }
    }
    candidates[lane] = vec2<f32>(best_error,f32(best_index));
}

fn refine_seed(reverse: bool, point: vec2<f32>) -> vec3<f32> {
    var best = vec2<f32>(1e30,289.0);
    for (var i=0u; i<64u; i++) {
        let candidate = candidates[i];
        // Equal errors use the first offset in row order, independent of lane.
        if candidate.x < best.x || (candidate.x == best.x && candidate.y < best.y) {
            best = candidate;
        }
    }
    if best.y >= 289.0 { return vec3<f32>(0.0); }
    let candidate = u32(best.y);
    var guess = search_initial+vec2<f32>(f32(candidate%17u)-8.0,f32(candidate/17u)-8.0);
    var origin = point;
    if reverse { origin = forward_result.xy; }
    var result = vec3<f32>(0.0);
    for (var index=i32(search_level); index>=0; index--) {
        let level = u32(index);
        let scale = f32(1u<<level);
        let p = (origin+vec2<f32>(0.5))/scale-vec2<f32>(0.5);
        let aligned = align(reverse,level,p,guess);
        if aligned.z == 0.0 { return vec3<f32>(0.0); }
        result = vec3<f32>((aligned.xy+vec2<f32>(0.5))*scale-vec2<f32>(0.5),1.0);
        guess = (result.xy+vec2<f32>(0.5))/(scale*0.5)-vec2<f32>(0.5);
    }
    return result;
}

// One workgroup shares the source patch and searches all 289 offsets.
@compute @workgroup_size(64)
fn match_points(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let id = group.x;
    if id >= params.control.x { return; }
    let point = corners[id];
    for (var direction=0u; direction<2u; direction++) {
        let reverse = direction == 1u;
        if lane == 0u {
            if direction == 0u { matches[id] = vec4<f32>(0.0); }
            prepare_search(reverse,point);
        }
        workgroupBarrier();
        search_candidates(reverse,lane);
        workgroupBarrier();
        if lane == 0u {
            let result = refine_seed(reverse,point);
            if direction == 0u { forward_result = result; }
            else if result.z > 0.0 && distance(result.xy,point) <= 1.0 {
                matches[id] = vec4<f32>(forward_result,0.0);
            }
        }
        workgroupBarrier();
    }
}
