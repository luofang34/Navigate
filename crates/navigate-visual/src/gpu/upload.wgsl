@group(0) @binding(0) var<storage,read> packed: array<u32>;
@group(0) @binding(1) var<storage,read_write> expanded: array<f32>;

@compute @workgroup_size(256)
fn unpack_gray(@builtin(global_invocation_id) global: vec3<u32>) {
    let start = global.x * 4u;
    let count = arrayLength(&expanded);
    if start >= count { return; }
    let value = packed[global.x];
    for (var byte=0u; byte<4u; byte++) {
        if start+byte < count {
            expanded[start+byte] = f32((value >> (byte*8u)) & 255u);
        }
    }
}
