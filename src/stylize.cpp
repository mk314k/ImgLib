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

} // namespace imglib
