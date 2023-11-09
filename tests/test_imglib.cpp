#include "imglib/imglib.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>

namespace {

bool approx_eq(float a, float b, float tol = 1e-4f) {
  return std::abs(a - b) <= tol;
}

void test_canvas_and_arithmetic() {
  imglib::Canvas a(16, 12, 3, 0.25f);
  imglib::Canvas b(16, 12, 3, 0.50f);
  assert(a.width() == 16 && a.height() == 12 && a.channels() == 3);

  imglib::Canvas sum = a + b;
  assert(approx_eq(sum.at(4, 5, 1), 0.75f));

  imglib::Canvas scaled = sum * 0.5f;
  assert(approx_eq(scaled.at(0, 0, 0), 0.375f));

  a.fill_rect(2, 2, 6, 6, 1.0f, 0.5f, 0.0f);
  assert(approx_eq(a.at(4, 4, 0), 1.0f));
  assert(approx_eq(a.at(4, 4, 1), 0.5f));
  assert(approx_eq(a.sample(-5, -5, 0, imglib::BorderMode::ZeroPad), 0.0f));
}

void test_png_and_ppm_io() {
  imglib::Canvas orig(10, 8, 3, 0.0f);
  orig.fill(0.2f, 0.6f, 0.8f);
  orig.stroke_line(0, 0, 9, 7, 1.0f, 1.0f, 0.0f);

  const std::string png_path = "/tmp/imglib_test_roundtrip.png";
  imglib::save_png(orig, png_path);
  imglib::Canvas decoded = imglib::load_png(png_path);
  assert(decoded.width() == 10 && decoded.height() == 8 && decoded.channels() == 3);
  assert(approx_eq(decoded.at(0, 0, 0), 1.0f, 0.01f));
  assert(approx_eq(decoded.at(8, 1, 2), 0.8f, 0.01f));
  std::remove(png_path.c_str());
}

void test_color_and_resampling() {
  imglib::Canvas rgb(12, 12, 3, 0.0f);
  rgb.fill(0.4f, 0.5f, 0.6f);

  imglib::Canvas yuv = imglib::rgb_to_yuv(rgb);
  imglib::Canvas back = imglib::yuv_to_rgb(yuv);
  assert(approx_eq(back.at(3, 3, 0), 0.4f, 1e-2f));
  assert(approx_eq(back.at(3, 3, 1), 0.5f, 1e-2f));
  assert(approx_eq(back.at(3, 3, 2), 0.6f, 1e-2f));

  // Verify black-pixel safety in split_luma_chroma
  imglib::Canvas black(4, 4, 3, 0.0f);
  imglib::LumaChromaPair lc = imglib::split_luma_chroma(black);
  imglib::Canvas merged = imglib::merge_luma_chroma(lc);
  assert(approx_eq(merged.at(1, 1, 0), 0.0f));

  imglib::Canvas up_lin = imglib::resize(rgb, 2.0f, imglib::FilterMode::Bilinear);
  imglib::Canvas up_cub = imglib::resize(rgb, 2.0f, imglib::FilterMode::Bicubic);
  imglib::Canvas up_lan = imglib::resize(rgb, 2.0f, imglib::FilterMode::Lanczos3);
  assert(up_lin.width() == 24 && up_cub.width() == 24 && up_lan.width() == 24);
}

void test_filters_and_bilateral() {
  imglib::Canvas imp = imglib::unit_impulse(9);
  imglib::Canvas g1 = imglib::gaussian_blur_separable(imp, 1.2f);
  imglib::Canvas g2 = imglib::gaussian_blur_2d(imp, 1.2f);
  assert(approx_eq(g1.at(4, 4, 0), g2.at(4, 4, 0), 1e-3f));

  imglib::Canvas step(16, 16, 3, 0.1f);
  step.fill_rect(8, 0, 15, 15, 0.9f, 0.9f, 0.9f);
  imglib::Canvas edges = imglib::sobel_magnitude(step);
  assert(edges.at(8, 8, 0) > 0.5f);

  imglib::Canvas den = imglib::bilateral_filter(step, 0.1f, 1.5f);
  assert(den.width() == 16 && den.height() == 16);
}

void test_warp_and_oriented_stylize() {
  imglib::Canvas a(20, 20, 3, 0.2f);
  imglib::Canvas b(20, 20, 3, 0.8f);
  std::vector<imglib::DirectedBone> s_bones = {
      imglib::DirectedBone(imglib::Vec2(4.0f, 4.0f), imglib::Vec2(16.0f, 4.0f))};
  std::vector<imglib::DirectedBone> d_bones = {
      imglib::DirectedBone(imglib::Vec2(4.0f, 6.0f), imglib::Vec2(16.0f, 6.0f))};

  std::vector<imglib::Canvas> seq = imglib::morph_sequence(a, b, s_bones, d_bones, 2);
  assert(seq.size() == 4u);

  const imglib::SymmetricEigen2x2 eig = imglib::solve_symmetric_2x2(4.0f, 0.0f, 1.0f);
  assert(approx_eq(eig.lambda_max, 4.0f));
  assert(approx_eq(eig.lambda_min, 1.0f));

  imglib::Canvas brush(7, 7, 1, 1.0f);
  imglib::Canvas painted =
      imglib::oriented_painterly_render(a, brush, 120, 6, 0.1f, 8, 7u);
  assert(painted.width() == 20 && painted.height() == 20);
}

} // namespace

int main() {
  test_canvas_and_arithmetic();
  test_png_and_ppm_io();
  test_color_and_resampling();
  test_filters_and_bilateral();
  test_warp_and_oriented_stylize();
  std::cout << "All ImgLib unit tests passed (5/5 suites).\n";
  return 0;
}
