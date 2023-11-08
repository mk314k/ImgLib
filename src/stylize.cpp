#include "imglib/stylize.hpp"
#include "imglib/color.hpp"
#include "imglib/filter.hpp"
#include "imglib/transform.hpp"
#include <cmath>
#include <random>

namespace imglib {

static constexpr float kTwoPi = 6.28318530717958647692f;

SymmetricEigen2x2 solve_symmetric_2x2(float j_xx, float j_xy, float j_yy) {
  const float trace = j_xx + j_yy;
  const float diff = j_xx - j_yy;
  const float disc = std::sqrt(diff * diff + 4.0f * j_xy * j_xy);
  const float l_min = 0.5f * (trace - disc);
  const float l_max = 0.5f * (trace + disc);

  // Principal gradient direction (max eigenvalue) and stroke tangent (min eigenvalue)
  float major_ang = 0.5f * std::atan2(2.0f * j_xy, diff);
  float minor_ang = major_ang + 0.5f * 3.14159265358979323846f;
  while (major_ang < 0.0f) major_ang += kTwoPi;
  while (minor_ang < 0.0f) minor_ang += kTwoPi;
  while (major_ang >= kTwoPi) major_ang -= kTwoPi;
  while (minor_ang >= kTwoPi) minor_ang -= kTwoPi;

  return {l_min, l_max, minor_ang, major_ang};
}

void splat_brush(Canvas& target,
                 int center_x,
                 int center_y,
                 const std::vector<float>& color,
                 const Canvas& mask) {
  const int half_w = mask.width() / 2;
  const int half_h = mask.height() / 2;

  for (int my = 0; my < mask.height(); ++my) {
    const int ty = center_y + my - half_h;
    if (ty < 0 || ty >= target.height()) continue;
    for (int mx = 0; mx < mask.width(); ++mx) {
      const int tx = center_x + mx - half_w;
      if (tx < 0 || tx >= target.width()) continue;
      const float alpha = std::max(0.0f, std::min(1.0f, mask.at(mx, my, 0)));
      for (int c = 0; c < target.channels(); ++c) {
        const float col_c =
            (static_cast<std::size_t>(c) < color.size()) ? color[static_cast<std::size_t>(c)] : 0.0f;
        target.at(tx, ty, c) =
            (1.0f - alpha) * target.at(tx, ty, c) + alpha * col_c;
      }
    }
  }
}

Canvas detail_importance_map(const Canvas& src, float sigma) {
  const Canvas luma = to_grayscale(src);
  const Canvas high_freq = luma - gaussian_blur_separable(luma, sigma);
  const Canvas energy = gaussian_blur_separable(high_freq * high_freq, 4.0f * sigma);
  const float peak = std::max(1e-6f, energy.max_value());
  return energy / peak;
}

Canvas structure_tensor(const Canvas& src, float sigma_grad, float sigma_window) {
  const Canvas luma = gaussian_blur_separable(to_grayscale(src), sigma_grad);
  const Canvas gx = sobel_x(luma);
  const Canvas gy = sobel_y(luma);

  Canvas raw_tensor(src.width(), src.height(), 3);
  for (int y = 0; y < src.height(); ++y) {
    for (int x = 0; x < src.width(); ++x) {
      const float ix = gx.at(x, y, 0);
      const float iy = gy.at(x, y, 0);
      raw_tensor.at(x, y, 0) = ix * ix;
      raw_tensor.at(x, y, 1) = ix * iy;
      raw_tensor.at(x, y, 2) = iy * iy;
    }
  }
  return gaussian_blur_separable(raw_tensor, sigma_grad * sigma_window);
}

Canvas orientation_field_map(const Canvas& src, float sigma_grad, float sigma_window) {
  const Canvas tensor = structure_tensor(src, sigma_grad, sigma_window);
  Canvas angle_map(src.width(), src.height(), 1);
  for (int y = 0; y < src.height(); ++y) {
    for (int x = 0; x < src.width(); ++x) {
      const SymmetricEigen2x2 eig = solve_symmetric_2x2(
          tensor.at(x, y, 0), tensor.at(x, y, 1), tensor.at(x, y, 2));
      angle_map.at(x, y, 0) = eig.minor_angle / kTwoPi;
    }
  }
  return angle_map;
}

static void apply_stroke_layer(const Canvas& src,
                               const Canvas& importance,
                               Canvas& canvas_out,
                               const Canvas& brush_mask,
                               int target_size,
                               int stroke_budget,
                               float color_jitter,
                               std::mt19937& rng) {
  const float scale = std::min(
      static_cast<float>(target_size) / static_cast<float>(brush_mask.width()),
      static_cast<float>(target_size) / static_cast<float>(brush_mask.height()));
  const Canvas scaled_brush =
      (scale < 1.0f) ? resize_bilinear(brush_mask, scale) : brush_mask;

  std::uniform_int_distribution<int> x_dist(0, src.width() - 1);
  std::uniform_int_distribution<int> y_dist(0, src.height() - 1);
  std::uniform_real_distribution<float> u01(0.0f, 1.0f);

  std::vector<float> color(static_cast<std::size_t>(src.channels()), 0.0f);
  int accepted = 0;
  int max_trials = stroke_budget * 5;
  for (int t = 0; t < max_trials && accepted < stroke_budget; ++t) {
    const int x = x_dist(rng);
    const int y = y_dist(rng);
    if (importance.at(x, y, 0) < u01(rng)) continue;
    for (int c = 0; c < src.channels(); ++c) {
      const float jitter = 1.0f - 0.5f * color_jitter + color_jitter * u01(rng);
      color[static_cast<std::size_t>(c)] =
          std::max(0.0f, std::min(1.0f, src.at(x, y, c) * jitter));
    }
    splat_brush(canvas_out, x, y, color, scaled_brush);
    ++accepted;
  }
}

Canvas painterly_render(const Canvas& src,
                        const Canvas& brush_mask,
                        int strokes,
                        int coarse_size,
                        float color_jitter,
                        unsigned int seed) {
  std::mt19937 rng(seed);
  Canvas out(src.width(), src.height(), src.channels(), 0.0f);
  const Canvas uniform_imp(src.width(), src.height(), 1, 1.0f);
  const Canvas detail_imp = detail_importance_map(src);

  apply_stroke_layer(src, uniform_imp, out, brush_mask, coarse_size, strokes,
                     color_jitter, rng);
  apply_stroke_layer(src, detail_imp, out, brush_mask,
                     std::max(4, coarse_size / 4), strokes, color_jitter, rng);
  return out;
}

Canvas oriented_painterly_render(const Canvas& src,
                                 const Canvas& brush_mask,
                                 int strokes,
                                 int coarse_size,
                                 float color_jitter,
                                 int angle_bins,
                                 unsigned int seed) {
  std::mt19937 rng(seed);
  Canvas out(src.width(), src.height(), src.channels(), 0.0f);
  const Canvas angles = orientation_field_map(src);
  const Canvas detail_imp = detail_importance_map(src);
  const Canvas uniform_imp(src.width(), src.height(), 1, 1.0f);

  auto paint_oriented_pass = [&](const Canvas& imp, int sz) {
    const float scale = std::min(
        static_cast<float>(sz) / static_cast<float>(brush_mask.width()),
        static_cast<float>(sz) / static_cast<float>(brush_mask.height()));
    const Canvas base_brush =
        (scale < 1.0f) ? resize_bilinear(brush_mask, scale) : brush_mask;

    std::vector<Canvas> rotated_bank;
    rotated_bank.reserve(static_cast<std::size_t>(angle_bins));
    for (int i = 0; i < angle_bins; ++i) {
      const float theta = kTwoPi * static_cast<float>(i) / static_cast<float>(angle_bins);
      rotated_bank.push_back(rotate_center(base_brush, theta));
    }

    std::uniform_int_distribution<int> x_dist(0, src.width() - 1);
    std::uniform_int_distribution<int> y_dist(0, src.height() - 1);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::vector<float> color(static_cast<std::size_t>(src.channels()), 0.0f);

    int accepted = 0;
    const int max_trials = strokes * 5;
    for (int t = 0; t < max_trials && accepted < strokes; ++t) {
      const int x = x_dist(rng);
      const int y = y_dist(rng);
      if (imp.at(x, y, 0) < u01(rng)) continue;
      for (int c = 0; c < src.channels(); ++c) {
        const float jitter = 1.0f - 0.5f * color_jitter + color_jitter * u01(rng);
        color[static_cast<std::size_t>(c)] =
            std::max(0.0f, std::min(1.0f, src.at(x, y, c) * jitter));
      }
      const int bin = std::min(
          angle_bins - 1,
          std::max(0, static_cast<int>(angles.at(x, y, 0) * angle_bins)));
      splat_brush(out, x, y, color, rotated_bank[static_cast<std::size_t>(bin)]);
      ++accepted;
    }
  };

  paint_oriented_pass(uniform_imp, coarse_size);
  paint_oriented_pass(detail_imp, std::max(4, coarse_size / 4));
  return out;
}

} // namespace imglib
