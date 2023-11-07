#ifndef IMGLIB_WARP_HPP
#define IMGLIB_WARP_HPP

#include "imglib/canvas.hpp"
#include <vector>

namespace imglib {

struct Vec2 {
  float x;
  float y;

  Vec2(float x_ = 0.0f, float y_ = 0.0f) : x(x_), y(y_) {}

  Vec2 operator+(const Vec2& o) const { return Vec2(x + o.x, y + o.y); }
  Vec2 operator-(const Vec2& o) const { return Vec2(x - o.x, y - o.y); }
  Vec2 operator*(float s) const { return Vec2(x * s, y * s); }
  Vec2 operator/(float s) const { return Vec2(x / s, y / s); }
};

float dot(const Vec2& a, const Vec2& b);
float norm(const Vec2& v);
Vec2 orthogonal(const Vec2& v);

class DirectedBone {
public:
  DirectedBone(Vec2 tail, Vec2 head);

  Vec2 to_local(Vec2 pt) const;
  Vec2 to_world(Vec2 uv) const;
  float point_distance(Vec2 pt) const;
  float field_weight(Vec2 pt, float a, float b, float p) const;

  const Vec2& tail() const { return tail_; }
  const Vec2& head() const { return head_; }
  float length() const { return len_; }

private:
  Vec2 tail_;
  Vec2 head_;
  float len_;
  Vec2 tangent_;
  Vec2 normal_;
};

Canvas warp_single_bone(const Canvas& src,
                        const DirectedBone& src_bone,
                        const DirectedBone& dst_bone);

Canvas warp_field(const Canvas& src,
                  const std::vector<DirectedBone>& src_bones,
                  const std::vector<DirectedBone>& dst_bones,
                  float a = 10.0f,
                  float b = 1.0f,
                  float p = 1.0f);

std::vector<Canvas> morph_sequence(const Canvas& start_img,
                                   const Canvas& end_img,
                                   const std::vector<DirectedBone>& start_bones,
                                   const std::vector<DirectedBone>& end_bones,
                                   int intermediate_frames = 3,
                                   float a = 10.0f,
                                   float b = 1.0f,
                                   float p = 1.0f);

} // namespace imglib

#endif // IMGLIB_WARP_HPP
