#include <metal_stdlib>
using namespace metal;

struct GemvShape {
    uint rows;
    uint columns;
};

inline ushort firewing_bf16(float value) {
    uint bits = as_type<uint>(value);
    if ((bits & 0x7f800000u) != 0x7f800000u) {
        bits += 0x7fffu + ((bits >> 16) & 1u);
    }
    return ushort(bits >> 16);
}

// Match the source-derived PyTorch aarch64 BF16 GEMV schedule: 32 F32
// accumulators, the 16/8/4 pairwise register tree, then (0+1)+(2+3).
kernel void firewing_bf16_gemv_exact(
    device const ushort *weights [[buffer(0)]],
    device const ushort *input [[buffer(1)]],
    device ushort *output [[buffer(2)]],
    constant GemvShape &shape [[buffer(3)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (row >= shape.rows || lane >= 32u) return;
    const uint row_offset = row * shape.columns;
    float sum = 0.0f;
    for (uint column = lane; column < shape.columns; column += 32u) {
        const float weight = as_type<float>(uint(weights[row_offset + column]) << 16);
        const float activation = as_type<float>(uint(input[column]) << 16);
        sum += weight * activation;
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 16u) partial[lane] += partial[lane + 16u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 8u) partial[lane] += partial[lane + 8u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 4u) partial[lane] += partial[lane + 4u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        output[row] = firewing_bf16((partial[0] + partial[1]) + (partial[2] + partial[3]));
    }
}
