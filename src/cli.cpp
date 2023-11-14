#include "imglib/imglib.hpp"
#include <iostream>
#include <string>

namespace {

imglib::Canvas make_disc_brush(int diameter) {
  imglib::Canvas mask(diameter, diameter, 1, 0.0f);
  const float r = 0.5f * static_cast<float>(diameter - 1);
  for (int y = 0; y < diameter; ++y) {
    for (int x = 0; x < diameter; ++x) {
      const float dx = static_cast<float>(x) - r;
      const float dy = static_cast<float>(y) - r;
      const float d = std::sqrt(dx * dx + dy * dy) / std::max(1.0f, r);
      if (d <= 1.0f) {
        mask.at(x, y, 0) = 1.0f - d * d;
      }
    }
  }
  return mask;
}

void print_usage() {
  std::cout
      << "Usage: imglib <command> <input.png> <output.png> [param]\n"
      << "Commands:\n"
      << "  grayscale <in.png> <out.png>\n"
      << "  gaussian  <in.png> <out.png> [sigma=2.0]\n"
      << "  sharpen   <in.png> <out.png> [strength=1.4]\n"
      << "  sobel     <in.png> <out.png>\n"
      << "  bilateral <in.png> <out.png> [sigma_spatial=2.5]\n"
      << "  resize    <in.png> <out.png> [scale=2.0]\n"
      << "  rotate    <in.png> <out.png> [angle_rad=0.52]\n"
      << "  stylize   <in.png> <out.png> [strokes=4000]\n"
      << "  equalize  <in.png> <out.png>\n"
      << "  hue       <in.png> <out.png> [delta_deg=30.0]\n"
      << "  canny     <in.png> <out.png> [sigma=1.2]\n"
      << "  median    <in.png> <out.png> [radius=1]\n"
      << "  vignette  <in.png> <out.png> [strength=0.45]\n";
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    print_usage();
    return 0;
  }

  const std::string cmd = argv[1];
  const std::string in_path = argv[2];
  const std::string out_path = argv[3];
  const float param = (argc >= 5) ? std::stof(argv[4]) : -1.0f;

  try {
    const imglib::Canvas src = imglib::load_png(in_path);
    imglib::Canvas result;

    if (cmd == "grayscale") {
      result = imglib::to_grayscale(src);
    } else if (cmd == "gaussian") {
      result = imglib::gaussian_blur_separable(src, (param > 0.0f) ? param : 2.0f);
    } else if (cmd == "sharpen") {
      result = imglib::unsharp_mask(src, 1.5f, (param > 0.0f) ? param : 1.4f);
    } else if (cmd == "sobel") {
      result = imglib::sobel_magnitude(src);
    } else if (cmd == "bilateral") {
      result = imglib::bilateral_yuv_filter(src, 0.1f, (param > 0.0f) ? param : 2.0f, 4.0f);
    } else if (cmd == "resize") {
      result = imglib::resize(src, (param > 0.0f) ? param : 2.0f,
                              imglib::FilterMode::Lanczos3);
    } else if (cmd == "rotate") {
      result = imglib::rotate_center(src, (param > 0.0f) ? param : 0.5235988f);
    } else if (cmd == "stylize") {
      const int strokes = (param > 0.0f) ? static_cast<int>(param) : 4000;
      result = imglib::oriented_painterly_render(src, make_disc_brush(15), strokes, 24);
    } else if (cmd == "equalize") {
      result = imglib::equalize_histogram(src);
    } else if (cmd == "hue") {
      result = imglib::shift_hsl(src, (param > 0.0f) ? param : 30.0f);
    } else if (cmd == "canny") {
      result = imglib::canny_edges(src, (param > 0.0f) ? param : 1.2f);
    } else if (cmd == "median") {
      const int rad = (param > 0.0f) ? static_cast<int>(param) : 1;
      result = imglib::median_filter(src, rad);
    } else if (cmd == "vignette") {
      result = imglib::apply_vignette(src, (param > 0.0f) ? param : 0.45f);
    } else {
      print_usage();
      return 1;
    }

    imglib::save_png(result, out_path);
  } catch (const std::exception& ex) {
    std::cerr << "imglib error: " << ex.what() << "\n";
    return 1;
  }
  return 0;
}
