#include "imglib/blend.hpp"
#include <algorithm>
#include <cmath>

namespace imglib {

float blend_channel(float b, float s, BlendMode mode) {
  const float cb = std::max(0.0f, std::min(1.0f, b));
  const float cs = std::max(0.0f, std::min(1.0f, s));

  switch (mode) {
    case BlendMode::Multiply:
      return cb * cs;
    case BlendMode::Screen:
      return 1.0f - (1.0f - cb) * (1.0f - cs);
    case BlendMode::Overlay:
      return (cb < 0.5f) ? (2.0f * cb * cs)
                         : (1.0f - 2.0f * (1.0f - cb) * (1.0f - cs));
    case BlendMode::SoftLight:
      return (1.0f - 2.0f * cs) * cb * cb + 2.0f * cs * cb;
    case BlendMode::HardLight:
      return (cs < 0.5f) ? (2.0f * cb * cs)
                         : (1.0f - 2.0f * (1.0f - cb) * (1.0f - cs));
    case BlendMode::Difference:
      return std::abs(cb - cs);
    case BlendMode::ColorDodge:
      if (cs >= 1.0f - 1e-6f) return 1.0f;
      return std::min(1.0f, cb / (1.0f - cs));
    case BlendMode::Normal:
    default:
      return cs;
  }
}

Canvas blend_layers(const Canvas& base,
                    const Canvas& top,
                    BlendMode mode,
                    float opacity) {
  ensure_same_shape(base, top);
  const float alpha = std::max(0.0f, std::min(1.0f, opacity));
  Canvas out(base.width(), base.height(), base.channels());

  for (int y = 0; y < base.height(); ++y) {
    for (int x = 0; x < base.width(); ++x) {
      for (int c = 0; c < base.channels(); ++c) {
        const float b_val = base.at(x, y, c);
        const float mixed = blend_channel(b_val, top.at(x, y, c), mode);
        out.at(x, y, c) = (1.0f - alpha) * b_val + alpha * mixed;
      }
    }
  }
  return out;
}

Canvas flatten_layer_stack(int width,
                           int height,
                           const std::vector<Layer>& layers,
                           float bg_r,
                           float bg_g,
                           float bg_b) {
  Canvas comp(width, height, 3, 0.0f);
  comp.fill(bg_r, bg_g, bg_b);

  for (const Layer& layer : layers) {
    const float alpha = std::max(0.0f, std::min(1.0f, layer.opacity));
    if (alpha <= 0.0f || layer.image.empty()) continue;

    for (int ly = 0; ly < layer.image.height(); ++ly) {
      const int ty = layer.offset_y + ly;
      if (ty < 0 || ty >= height) continue;
      for (int lx = 0; lx < layer.image.width(); ++lx) {
        const int tx = layer.offset_x + lx;
        if (tx < 0 || tx >= width) continue;
        for (int c = 0; c < 3; ++c) {
          const int src_c = std::min(c, layer.image.channels() - 1);
          const float b_val = comp.at(tx, ty, c);
          const float s_val = layer.image.at(lx, ly, src_c);
          const float mixed = blend_channel(b_val, s_val, layer.mode);
          comp.at(tx, ty, c) = (1.0f - alpha) * b_val + alpha * mixed;
        }
      }
    }
  }
  return comp;
}

Canvas radial_feather_mask(int width,
                           int height,
                           float center_x,
                           float center_y,
                           float inner_radius,
                           float outer_radius) {
  Canvas mask(width, height, 1, 0.0f);
  const float r_in = std::max(0.0f, inner_radius);
  const float r_out = std::max(r_in + 1e-4f, outer_radius);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const float dx = static_cast<float>(x) - center_x;
      const float dy = static_cast<float>(y) - center_y;
      const float dist = std::sqrt(dx * dx + dy * dy);
      if (dist <= r_in) {
        mask.at(x, y, 0) = 1.0f;
      } else if (dist < r_out) {
        const float t = (dist - r_in) / (r_out - r_in);
        mask.at(x, y, 0) = 1.0f - t * t * (3.0f - 2.0f * t);
      } else {
        mask.at(x, y, 0) = 0.0f;
      }
    }
  }
  return mask;
}

Canvas masked_blend(const Canvas& base,
                    const Canvas& top,
                    const Canvas& mask,
                    BlendMode mode) {
  ensure_same_shape(base, top);
  if (mask.width() != base.width() || mask.height() != base.height()) {
    throw std::invalid_argument("masked_blend mask dimensions must match base");
  }
  Canvas out(base.width(), base.height(), base.channels());

  for (int y = 0; y < base.height(); ++y) {
    for (int x = 0; x < base.width(); ++x) {
      const float m = std::max(0.0f, std::min(1.0f, mask.at(x, y, 0)));
      for (int c = 0; c < base.channels(); ++c) {
        const float b_val = base.at(x, y, c);
        const float mixed = blend_channel(b_val, top.at(x, y, c), mode);
        out.at(x, y, c) = (1.0f - m) * b_val + m * mixed;
      }
    }
  }
  return out;
}

} // namespace imglib
