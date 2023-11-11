#ifndef IMGLIB_BLEND_HPP
#define IMGLIB_BLEND_HPP

#include "imglib/canvas.hpp"
#include <vector>

namespace imglib {

enum class BlendMode {
  Normal,
  Multiply,
  Screen,
  Overlay,
  SoftLight,
  HardLight,
  Difference,
  ColorDodge
};

struct Layer {
  Canvas image;
  BlendMode mode = BlendMode::Normal;
  float opacity = 1.0f;
  int offset_x = 0;
  int offset_y = 0;
};

float blend_channel(float base, float top, BlendMode mode);

Canvas blend_layers(const Canvas& base,
                    const Canvas& top,
                    BlendMode mode = BlendMode::Normal,
                    float opacity = 1.0f);

Canvas flatten_layer_stack(int width,
                           int height,
                           const std::vector<Layer>& layers,
                           float bg_r = 0.0f,
                           float bg_g = 0.0f,
                           float bg_b = 0.0f);

} // namespace imglib

#endif // IMGLIB_BLEND_HPP
