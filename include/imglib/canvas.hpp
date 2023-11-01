#ifndef IMGLIB_CANVAS_HPP
#define IMGLIB_CANVAS_HPP

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace imglib {

enum class BorderMode {
  ZeroPad,
  ClampEdge,
  Reflect
};

class Canvas {
public:
  Canvas() : width_(0), height_(0), channels_(0) {}
  Canvas(int width, int height, int channels = 3, float init_val = 0.0f);

  int width() const { return width_; }
  int height() const { return height_; }
  int channels() const { return channels_; }
  std::size_t size() const { return buffer_.size(); }
  bool empty() const { return buffer_.empty(); }

  float* data() { return buffer_.data(); }
  const float* data() const { return buffer_.data(); }

  float& at(int x, int y, int c = 0);
  const float& at(int x, int y, int c = 0) const;

  float& operator()(int x, int y, int c = 0) { return at(x, y, c); }
  const float& operator()(int x, int y, int c = 0) const { return at(x, y, c); }

  float& flat(std::size_t idx);
  const float& flat(std::size_t idx) const;

  float sample(int x, int y, int c = 0,
               BorderMode mode = BorderMode::ClampEdge,
               float pad_val = 0.0f) const;

  float min_value() const;
  float max_value() const;
  Canvas clamped(float lo = 0.0f, float hi = 1.0f) const;

  void fill(float r, float g = 0.0f, float b = 0.0f);
  void fill_rect(int x0, int y0, int x1, int y1,
                 float r, float g = 0.0f, float b = 0.0f);
  void stroke_line(int x0, int y0, int x1, int y1,
                   float r, float g = 0.0f, float b = 0.0f);

private:
  int width_;
  int height_;
  int channels_;
  std::vector<float> buffer_;

  std::size_t index(int x, int y, int c) const {
    return static_cast<std::size_t>((y * width_ + x) * channels_ + c);
  }
};

void ensure_same_shape(const Canvas& a, const Canvas& b);

Canvas operator+(const Canvas& a, const Canvas& b);
Canvas operator-(const Canvas& a, const Canvas& b);
Canvas operator*(const Canvas& a, const Canvas& b);
Canvas operator/(const Canvas& a, const Canvas& b);

Canvas operator+(const Canvas& a, float scalar);
Canvas operator-(const Canvas& a, float scalar);
Canvas operator*(const Canvas& a, float scalar);
Canvas operator/(const Canvas& a, float scalar);

Canvas operator+(float scalar, const Canvas& a);
Canvas operator-(float scalar, const Canvas& a);
Canvas operator*(float scalar, const Canvas& a);
Canvas operator/(float scalar, const Canvas& a);

} // namespace imglib

#endif // IMGLIB_CANVAS_HPP
