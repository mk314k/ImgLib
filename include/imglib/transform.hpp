#ifndef IMGLIB_TRANSFORM_HPP
#define IMGLIB_TRANSFORM_HPP

#include "imglib/canvas.hpp"

namespace imglib {

enum class FilterMode {
  Nearest,
  Bilinear,
  Bicubic,
  Lanczos3
};

float sample_bilinear(const Canvas& src,
                      float x,
                      float y,
                      int c,
                      BorderMode border = BorderMode::ClampEdge);

Canvas resize_nearest(const Canvas& src, float scale_x, float scale_y = -1.0f);
Canvas resize_bilinear(const Canvas& src, float scale_x, float scale_y = -1.0f);
Canvas resize_bicubic(const Canvas& src,
                      float scale_x,
                      float scale_y = -1.0f,
                      float b_param = 1.0f / 3.0f,
                      float c_param = 1.0f / 3.0f);
Canvas resize_lanczos(const Canvas& src,
                      float scale_x,
                      float scale_y = -1.0f,
                      float lobes = 3.0f);

Canvas resize(const Canvas& src,
              float scale,
              FilterMode mode = FilterMode::Bilinear);

Canvas rotate_center(const Canvas& src, float angle_radians);

Canvas crop_region(const Canvas& src, int x0, int y0, int width, int height);
Canvas flip_horizontal(const Canvas& src);
Canvas flip_vertical(const Canvas& src);
Canvas apply_vignette(const Canvas& src, float strength = 0.45f, float power = 1.8f);

} // namespace imglib

#endif // IMGLIB_TRANSFORM_HPP
