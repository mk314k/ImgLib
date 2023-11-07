#ifndef IMGLIB_STYLIZE_HPP
#define IMGLIB_STYLIZE_HPP

#include "imglib/canvas.hpp"
#include <vector>

namespace imglib {

struct SymmetricEigen2x2 {
  float lambda_min;
  float lambda_max;
  float minor_angle;
  float major_angle;
};

SymmetricEigen2x2 solve_symmetric_2x2(float j_xx, float j_xy, float j_yy);

void splat_brush(Canvas& target,
                 int center_x,
                 int center_y,
                 const std::vector<float>& color,
                 const Canvas& mask);

Canvas detail_importance_map(const Canvas& src, float sigma = 1.0f);

Canvas structure_tensor(const Canvas& src,
                        float sigma_grad = 1.0f,
                        float sigma_window = 4.0f);

Canvas orientation_field_map(const Canvas& src,
                             float sigma_grad = 1.5f,
                             float sigma_window = 4.0f);

Canvas painterly_render(const Canvas& src,
                        const Canvas& brush_mask,
                        int strokes = 6000,
                        int coarse_size = 36,
                        float color_jitter = 0.15f,
                        unsigned int seed = 42u);

Canvas oriented_painterly_render(const Canvas& src,
                                 const Canvas& brush_mask,
                                 int strokes = 6000,
                                 int coarse_size = 36,
                                 float color_jitter = 0.15f,
                                 int angle_bins = 24,
                                 unsigned int seed = 42u);

} // namespace imglib

#endif // IMGLIB_STYLIZE_HPP
