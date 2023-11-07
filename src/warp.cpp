#include "imglib/warp.hpp"
#include "imglib/transform.hpp"
#include <cmath>

namespace imglib {

float dot(const Vec2& a, const Vec2& b) {
  return a.x * b.x + a.y * b.y;
}

float norm(const Vec2& v) {
  return std::sqrt(v.x * v.x + v.y * v.y);
}

Vec2 orthogonal(const Vec2& v) {
  return Vec2(-v.y, v.x);
}

DirectedBone::DirectedBone(Vec2 tail, Vec2 head)
    : tail_(tail), head_(head) {
  const float d = norm(head_ - tail_);
  len_ = (d > 1e-6f) ? d : 1e-6f;
  tangent_ = (head_ - tail_) / len_;
  normal_ = orthogonal(tangent_);
}

Vec2 DirectedBone::to_local(Vec2 pt) const {
  const Vec2 rel = pt - tail_;
  return Vec2(dot(rel, tangent_), dot(rel, normal_));
}

Vec2 DirectedBone::to_world(Vec2 uv) const {
  return tail_ + tangent_ * uv.x + normal_ * uv.y;
}

float DirectedBone::point_distance(Vec2 pt) const {
  const Vec2 uv = to_local(pt);
  if (uv.x < 0.0f) return norm(pt - tail_);
  if (uv.x > len_) return norm(pt - head_);
  return std::abs(uv.y);
}

float DirectedBone::field_weight(Vec2 pt, float a, float b, float p) const {
  const float dist = point_distance(pt);
  return std::pow(std::pow(len_, p) / (a + dist), b);
}

} // namespace imglib
