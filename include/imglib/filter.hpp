#ifndef IMGLIB_FILTER_HPP
#define IMGLIB_FILTER_HPP

#include "imglib/canvas.hpp"
#include <vector>

namespace imglib {

class Kernel2D {
public:
  Kernel2D(int width, int height, float fill_val = 0.0f);
  Kernel2D(int width, int height, const std::vector<float>& weights);

  int width() const { return width_; }
  int height() const { return height_; }

  float& operator()(int x, int y);
  const float& operator()(int x, int y) const;

  Kernel2D transposed() const;
  Canvas convolve(const Canvas& src, BorderMode border = BorderMode::ClampEdge) const;

private:
  int width_;
  int height_;
  std::vector<float> weights_;
};

std::vector<float> gaussian_kernel_1d(float sigma, float truncate = 3.0f);
std::vector<float> gaussian_kernel_2d(float sigma, float truncate = 3.0f);

Canvas box_blur(const Canvas& src, int radius, BorderMode border = BorderMode::ClampEdge);
Canvas gaussian_blur_separable(const Canvas& src,
                               float sigma,
                               float truncate = 3.0f,
                               BorderMode border = BorderMode::ClampEdge);
Canvas gaussian_blur_2d(const Canvas& src,
                        float sigma,
                        float truncate = 3.0f,
                        BorderMode border = BorderMode::ClampEdge);

Canvas unsharp_mask(const Canvas& src,
                    float sigma = 1.5f,
                    float strength = 1.2f,
                    float truncate = 3.0f);

Canvas sobel_x(const Canvas& src, BorderMode border = BorderMode::ClampEdge);
Canvas sobel_y(const Canvas& src, BorderMode border = BorderMode::ClampEdge);
Canvas sobel_magnitude(const Canvas& src, BorderMode border = BorderMode::ClampEdge);

Canvas bilateral_filter(const Canvas& src,
                        float sigma_range = 0.1f,
                        float sigma_spatial = 2.0f,
                        float truncate = 3.0f,
                        BorderMode border = BorderMode::ClampEdge);

Canvas bilateral_yuv_filter(const Canvas& src,
                            float sigma_range = 0.1f,
                            float sigma_y = 1.5f,
                            float sigma_uv = 4.0f,
                            float truncate = 3.0f);

Canvas max_pool_filter(const Canvas& src, int window_diameter);
Canvas unit_impulse(int size);

} // namespace imglib

#endif // IMGLIB_FILTER_HPP
