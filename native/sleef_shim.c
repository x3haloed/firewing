#include <arm_neon.h>
#include <Accelerate/Accelerate.h>
#include <stddef.h>
#include <stdlib.h>
#include <sleef.h>

typedef float32x4_t (*firewing_sleef_unary)(float32x4_t);

static void firewing_sleef_map(float *output, const float *input, size_t count,
                               firewing_sleef_unary function) {
  size_t offset = 0;
  for (; offset + 4 <= count; offset += 4) {
    vst1q_f32(output + offset, function(vld1q_f32(input + offset)));
  }
  if (offset < count) {
    float tail_input[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float tail_output[4];
    size_t tail = count - offset;
    for (size_t index = 0; index < tail; ++index) {
      tail_input[index] = input[offset + index];
    }
    vst1q_f32(tail_output, function(vld1q_f32(tail_input)));
    for (size_t index = 0; index < tail; ++index) {
      output[offset + index] = tail_output[index];
    }
  }
}

void firewing_sleef_expf_u10(float *output, const float *input, size_t count) {
  firewing_sleef_map(output, input, count, Sleef_expf4_u10);
}

void firewing_sleef_log1pf_u10(float *output, const float *input, size_t count) {
  firewing_sleef_map(output, input, count, Sleef_log1pf4_u10);
}

static float32x4_t firewing_neon_sqrt(float32x4_t input) {
  return vsqrtq_f32(input);
}

static float32x4_t firewing_neon_rsqrt(float32x4_t input) {
  return vdivq_f32(vdupq_n_f32(1.0f), vsqrtq_f32(input));
}

void firewing_neon_rsqrtf(float *output, const float *input, size_t count) {
  firewing_sleef_map(output, input, count, firewing_neon_rsqrt);
}

static float32x4_t firewing_sleef_sigmoid(float32x4_t input) {
  float32x4_t denominator =
      vaddq_f32(vdupq_n_f32(1.0f), Sleef_expf4_u10(vnegq_f32(input)));
  return vdivq_f32(vdupq_n_f32(1.0f), denominator);
}

void firewing_sleef_sigmoidf(float *output, const float *input, size_t count) {
  firewing_sleef_map(output, input, count, firewing_sleef_sigmoid);
}

void firewing_neon_sqrtf(float *output, const float *input, size_t count) {
  firewing_sleef_map(output, input, count, firewing_neon_sqrt);
}

static float32x4_t firewing_neon_reciprocal(float32x4_t input) {
  return vdivq_f32(vdupq_n_f32(1.0f), input);
}

void firewing_neon_reciprocalf(float *output, const float *input, size_t count) {
  firewing_sleef_map(output, input, count, firewing_neon_reciprocal);
}

float firewing_accelerate_padded_dot(const float *left, const float *right) {
  float *matrix_left = calloc(64 * 128, sizeof(float));
  float *matrix_right = calloc(128 * 64, sizeof(float));
  float *matrix_output = calloc(64 * 64, sizeof(float));
  if (matrix_left == NULL || matrix_right == NULL || matrix_output == NULL) {
    free(matrix_left);
    free(matrix_right);
    free(matrix_output);
    return 0.0f / 0.0f;
  }
  for (size_t index = 0; index < 128; ++index) {
    matrix_left[index] = left[index];
    matrix_right[index * 64] = right[index];
  }
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64, 64, 128, 1.0f,
              matrix_left, 128, matrix_right, 64, 0.0f, matrix_output, 64);
  float result = matrix_output[0];
  free(matrix_left);
  free(matrix_right);
  free(matrix_output);
  return result;
}
