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

Canvas warp_single_bone(const Canvas& src,
                        const DirectedBone& src_bone,
                        const DirectedBone& dst_bone) {
  Canvas out(src.width(), src.height(), src.channels());
  for (int y = 0; y < out.height(); ++y) {
    for (int x = 0; x < out.width(); ++x) {
      const Vec2 dst_pt(static_cast<float>(x), static_cast<float>(y));
      const Vec2 uv = dst_bone.to_local(dst_pt);
      const Vec2 src_pt = src_bone.to_world(uv);
      for (int c = 0; c < out.channels(); ++c) {
        out.at(x, y, c) = sample_bilinear(src, src_pt.x, src_pt.y, c, BorderMode::ClampEdge);
      }
    }
  }
  return out;
}

Canvas warp_field(const Canvas& src,
                  const std::vector<DirectedBone>& src_bones,
                  const std::vector<DirectedBone>& dst_bones,
                  float a,
                  float b,
                  float p) {
  if (src_bones.size() != dst_bones.size() || src_bones.empty()) {
    throw std::invalid_argument("warp_field requires matching non-empty bone sets");
  }
  Canvas out(src.width(), src.height(), src.channels());

  for (int y = 0; y < out.height(); ++y) {
    for (int x = 0; x < out.width(); ++x) {
      const Vec2 dst_pt(static_cast<float>(x), static_cast<float>(y));
      Vec2 disp_sum(0.0f, 0.0f);
      float weight_sum = 0.0f;

      for (std::size_t i = 0; i < dst_bones.size(); ++i) {
        const Vec2 uv = dst_bones[i].to_local(dst_pt);
        const Vec2 mapped = src_bones[i].to_world(uv);
        const float w = dst_bones[i].field_weight(dst_pt, a, b, p);
        disp_sum = disp_sum + (mapped - dst_pt) * w;
        weight_sum += w;
      }
      const Vec2 src_pt = dst_pt + disp_sum / std::max(1e-8f, weight_sum);
      for (int c = 0; c < out.channels(); ++c) {
        out.at(x, y, c) = sample_bilinear(src, src_pt.x, src_pt.y, c, BorderMode::ClampEdge);
      }
    }
  }
  return out;
}

std::vector<Canvas> morph_sequence(const Canvas& start_img,
                                   const Canvas& end_img,
                                   const std::vector<DirectedBone>& start_bones,
                                   const std::vector<DirectedBone>& end_bones,
                                   int intermediate_frames,
                                   float a,
                                   float b,
                                   float p) {
  ensure_same_shape(start_img, end_img);
  if (start_bones.size() != end_bones.size()) {
    throw std::invalid_argument("morph_sequence requires matching bone counts");
  }

  std::vector<Canvas> frames;
  frames.reserve(static_cast<std::size_t>(intermediate_frames + 2));
  frames.push_back(start_img);

  for (int f = 1; f <= intermediate_frames; ++f) {
    const float t = static_cast<float>(f) / static_cast<float>(intermediate_frames + 1);
    std::vector<DirectedBone> mid_bones;
    mid_bones.reserve(start_bones.size());

    for (std::size_t j = 0; j < start_bones.size(); ++j) {
      const Vec2 p_interp =
          start_bones[j].tail() * (1.0f - t) + end_bones[j].tail() * t;
      const Vec2 q_interp =
          start_bones[j].head() * (1.0f - t) + end_bones[j].head() * t;
      mid_bones.emplace_back(p_interp, q_interp);
    }

    const Canvas warped_start = warp_field(start_img, start_bones, mid_bones, a, b, p);
    const Canvas warped_end = warp_field(end_img, end_bones, mid_bones, a, b, p);
    frames.push_back(warped_start * (1.0f - t) + warped_end * t);
  }

  frames.push_back(end_img);
  return frames;
}

} // namespace imglib
