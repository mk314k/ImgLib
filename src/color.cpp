#include "imglib/color.hpp"
#include <cmath>

namespace imglib {

Canvas adjust_brightness(const Canvas& src, float gain) {
  return src * gain;
}

Canvas adjust_contrast(const Canvas& src, float factor, float pivot) {
  return (src - pivot) * factor + pivot;
}

Canvas to_grayscale(const Canvas& src, float wr, float wg, float wb) {
  if (src.channels() == 1) return src;
  const float norm = (wr + wg + wb != 0.0f) ? (wr + wg + wb) : 1.0f;
  Canvas gray(src.width(), src.height(), 1);
  for (int y = 0; y < src.height(); ++y) {
    for (int x = 0; x < src.width(); ++x) {
      const float val =
          (src.at(x, y, 0) * wr + src.at(x, y, 1) * wg + src.at(x, y, 2) * wb) /
          norm;
      gray.at(x, y, 0) = val;
    }
  }
  return gray;
}

LumaChromaPair split_luma_chroma(const Canvas& src) {
  Canvas luma = to_grayscale(src);
  Canvas chroma(src.width(), src.height(), src.channels());
  for (int y = 0; y < src.height(); ++y) {
    for (int x = 0; x < src.width(); ++x) {
      const float l_val = luma.at(x, y, 0);
      const float safe_l = (l_val > 1e-6f) ? l_val : 1e-6f;
      for (int c = 0; c < src.channels(); ++c) {
        chroma.at(x, y, c) = (l_val > 1e-6f) ? (src.at(x, y, c) / safe_l) : 0.0f;
      }
    }
  }
  return {luma, chroma};
}

Canvas merge_luma_chroma(const LumaChromaPair& pair) {
  Canvas out(pair.chroma.width(), pair.chroma.height(), pair.chroma.channels());
  for (int y = 0; y < out.height(); ++y) {
    for (int x = 0; x < out.width(); ++x) {
      const float l_val = pair.luma.at(x, y, 0);
      for (int c = 0; c < out.channels(); ++c) {
        out.at(x, y, c) = pair.chroma.at(x, y, c) * l_val;
      }
    }
  }
  return out;
}

Canvas adjust_luma_tone(const Canvas& src,
                        float brightness_gain,
                        float contrast_gain,
                        float pivot) {
  LumaChromaPair lc = split_luma_chroma(src);
  lc.luma = adjust_brightness(lc.luma, brightness_gain);
  lc.luma = adjust_contrast(lc.luma, contrast_gain, pivot);
  return merge_luma_chroma(lc);
}

Canvas rgb_to_yuv(const Canvas& src) {
  if (src.channels() < 3) {
    throw std::invalid_argument("rgb_to_yuv requires at least 3 channels");
  }
  Canvas out(src.width(), src.height(), 3);
  for (int y = 0; y < src.height(); ++y) {
    for (int x = 0; x < src.width(); ++x) {
      const float r = src.at(x, y, 0);
      const float g = src.at(x, y, 1);
      const float b = src.at(x, y, 2);
      out.at(x, y, 0) = 0.299f * r + 0.587f * g + 0.114f * b;
      out.at(x, y, 1) = -0.147f * r - 0.289f * g + 0.436f * b;
      out.at(x, y, 2) = 0.615f * r - 0.515f * g - 0.100f * b;
    }
  }
  return out;
}

Canvas yuv_to_rgb(const Canvas& src) {
  if (src.channels() < 3) {
    throw std::invalid_argument("yuv_to_rgb requires at least 3 channels");
  }
  Canvas out(src.width(), src.height(), 3);
  for (int y = 0; y < src.height(); ++y) {
    for (int x = 0; x < src.width(); ++x) {
      const float y_val = src.at(x, y, 0);
      const float u_val = src.at(x, y, 1);
      const float v_val = src.at(x, y, 2);
      out.at(x, y, 0) = y_val + 1.140f * v_val;
      out.at(x, y, 1) = y_val - 0.395f * u_val - 0.581f * v_val;
      out.at(x, y, 2) = y_val + 2.032f * u_val;
    }
  }
  return out;
}

Canvas adjust_saturation(const Canvas& src, float scale) {
  Canvas yuv = rgb_to_yuv(src);
  for (int y = 0; y < yuv.height(); ++y) {
    for (int x = 0; x < yuv.width(); ++x) {
      yuv.at(x, y, 1) *= scale;
      yuv.at(x, y, 2) *= scale;
    }
  }
  return yuv_to_rgb(yuv);
}

Canvas apply_gamma(const Canvas& src, float gamma) {
  if (gamma <= 0.0f) {
    throw std::invalid_argument("Gamma must be positive");
  }
  const float inv_gamma = 1.0f / gamma;
  Canvas out(src.width(), src.height(), src.channels());
  for (std::size_t i = 0; i < src.size(); ++i) {
    const float v = std::max(0.0f, src.flat(i));
    out.flat(i) = std::pow(v, inv_gamma);
  }
  return out;
}

Canvas quantize_bits(const Canvas& src, int bits_per_channel) {
  if (bits_per_channel < 1 || bits_per_channel > 16) {
    throw std::invalid_argument("bits_per_channel must be in [1, 16]");
  }
  const float steps = std::pow(2.0f, static_cast<float>(bits_per_channel)) - 1.0f;
  Canvas out(src.width(), src.height(), src.channels());
  for (std::size_t i = 0; i < src.size(); ++i) {
    const float clamped = std::max(0.0f, std::min(1.0f, src.flat(i)));
    out.flat(i) = std::round(clamped * steps) / steps;
  }
  return out;
}

Canvas gray_world_balance(const Canvas& src) {
  if (src.channels() < 3) return src;
  double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
  for (int y = 0; y < src.height(); ++y) {
    for (int x = 0; x < src.width(); ++x) {
      sum_r += src.at(x, y, 0);
      sum_g += src.at(x, y, 1);
      sum_b += src.at(x, y, 2);
    }
  }
  const float gain_r = (sum_r > 1e-9) ? static_cast<float>(sum_g / sum_r) : 1.0f;
  const float gain_b = (sum_b > 1e-9) ? static_cast<float>(sum_g / sum_b) : 1.0f;

  Canvas out(src.width(), src.height(), src.channels());
  for (int y = 0; y < src.height(); ++y) {
    for (int x = 0; x < src.width(); ++x) {
      out.at(x, y, 0) = src.at(x, y, 0) * gain_r;
      out.at(x, y, 1) = src.at(x, y, 1);
      out.at(x, y, 2) = src.at(x, y, 2) * gain_b;
    }
  }
  return out;
}

std::pair<Canvas, Canvas> opponent_afterimage_pair(const Canvas& src) {
  Canvas luma = to_grayscale(src);
  Canvas inv_chroma = rgb_to_yuv(src);
  for (int y = 0; y < inv_chroma.height(); ++y) {
    for (int x = 0; x < inv_chroma.width(); ++x) {
      inv_chroma.at(x, y, 0) = 0.5f;
      inv_chroma.at(x, y, 1) = -inv_chroma.at(x, y, 1);
      inv_chroma.at(x, y, 2) = -inv_chroma.at(x, y, 2);
    }
  }
  inv_chroma = yuv_to_rgb(inv_chroma);
  const int cx = src.width() / 2;
  const int cy = src.height() / 2;
  luma.at(cx, cy, 0) = 0.0f;
  for (int c = 0; c < 3; ++c) inv_chroma.at(cx, cy, c) = 0.0f;
  return {inv_chroma, luma};
}

Canvas rgb_to_hsl(const Canvas& src) {
  if (src.channels() < 3) {
    throw std::invalid_argument("rgb_to_hsl requires 3 channels");
  }
  Canvas out(src.width(), src.height(), 3);
  for (int y = 0; y < src.height(); ++y) {
    for (int x = 0; x < src.width(); ++x) {
      const float r = std::max(0.0f, std::min(1.0f, src.at(x, y, 0)));
      const float g = std::max(0.0f, std::min(1.0f, src.at(x, y, 1)));
      const float b = std::max(0.0f, std::min(1.0f, src.at(x, y, 2)));
      const float cmax = std::max(r, std::max(g, b));
      const float cmin = std::min(r, std::min(g, b));
      const float delta = cmax - cmin;
      const float l = 0.5f * (cmax + cmin);

      float h = 0.0f;
      float s = 0.0f;
      if (delta > 1e-6f) {
        s = delta / (1.0f - std::abs(2.0f * l - 1.0f));
        if (cmax == r) {
          h = std::fmod((g - b) / delta, 6.0f);
          if (h < 0.0f) h += 6.0f;
        } else if (cmax == g) {
          h = ((b - r) / delta) + 2.0f;
        } else {
          h = ((r - g) / delta) + 4.0f;
        }
        h /= 6.0f;
      }
      out.at(x, y, 0) = h;
      out.at(x, y, 1) = std::max(0.0f, std::min(1.0f, s));
      out.at(x, y, 2) = std::max(0.0f, std::min(1.0f, l));
    }
  }
  return out;
}

Canvas hsl_to_rgb(const Canvas& src) {
  if (src.channels() < 3) {
    throw std::invalid_argument("hsl_to_rgb requires 3 channels");
  }
  Canvas out(src.width(), src.height(), 3);
  for (int y = 0; y < src.height(); ++y) {
    for (int x = 0; x < src.width(); ++x) {
      float h = std::fmod(src.at(x, y, 0), 1.0f);
      if (h < 0.0f) h += 1.0f;
      const float s = std::max(0.0f, std::min(1.0f, src.at(x, y, 1)));
      const float l = std::max(0.0f, std::min(1.0f, src.at(x, y, 2)));

      const float c = (1.0f - std::abs(2.0f * l - 1.0f)) * s;
      const float h6 = h * 6.0f;
      const float x_val = c * (1.0f - std::abs(std::fmod(h6, 2.0f) - 1.0f));
      const float m = l - 0.5f * c;

      float r1 = 0.0f, g1 = 0.0f, b1 = 0.0f;
      if (h6 < 1.0f) {
        r1 = c; g1 = x_val; b1 = 0.0f;
      } else if (h6 < 2.0f) {
        r1 = x_val; g1 = c; b1 = 0.0f;
      } else if (h6 < 3.0f) {
        r1 = 0.0f; g1 = c; b1 = x_val;
      } else if (h6 < 4.0f) {
        r1 = 0.0f; g1 = x_val; b1 = c;
      } else if (h6 < 5.0f) {
        r1 = x_val; g1 = 0.0f; b1 = c;
      } else {
        r1 = c; g1 = 0.0f; b1 = x_val;
      }
      out.at(x, y, 0) = r1 + m;
      out.at(x, y, 1) = g1 + m;
      out.at(x, y, 2) = b1 + m;
    }
  }
  return out;
}

Canvas shift_hsl(const Canvas& src,
                 float delta_hue_deg,
                 float sat_scale,
                 float light_offset) {
  Canvas hsl = rgb_to_hsl(src);
  const float dh = delta_hue_deg / 360.0f;
  for (int y = 0; y < hsl.height(); ++y) {
    for (int x = 0; x < hsl.width(); ++x) {
      float h = std::fmod(hsl.at(x, y, 0) + dh, 1.0f);
      if (h < 0.0f) h += 1.0f;
      hsl.at(x, y, 0) = h;
      hsl.at(x, y, 1) = std::max(0.0f, std::min(1.0f, hsl.at(x, y, 1) * sat_scale));
      hsl.at(x, y, 2) = std::max(0.0f, std::min(1.0f, hsl.at(x, y, 2) + light_offset));
    }
  }
  return hsl_to_rgb(hsl);
}

Canvas adjust_levels(const Canvas& src,
                     float in_black,
                     float in_white,
                     float gamma,
                     float out_black,
                     float out_white) {
  const float span = std::max(1e-6f, in_white - in_black);
  const float inv_gamma = 1.0f / std::max(1e-4f, gamma);
  Canvas out(src.width(), src.height(), src.channels());
  for (std::size_t i = 0; i < src.size(); ++i) {
    const float norm =
        std::max(0.0f, std::min(1.0f, (src.flat(i) - in_black) / span));
    const float curved = std::pow(norm, inv_gamma);
    out.flat(i) = out_black + curved * (out_white - out_black);
  }
  return out;
}

Canvas equalize_histogram(const Canvas& src) {
  LumaChromaPair lc = split_luma_chroma(src);
  std::vector<int> hist(256, 0);
  const int total_px = lc.luma.width() * lc.luma.height();

  for (int y = 0; y < lc.luma.height(); ++y) {
    for (int x = 0; x < lc.luma.width(); ++x) {
      const int bin = std::max(
          0, std::min(255, static_cast<int>(std::round(lc.luma.at(x, y, 0) * 255.0f))));
      hist[static_cast<std::size_t>(bin)] += 1;
    }
  }

  std::vector<float> cdf(256, 0.0f);
  int running = 0;
  for (int b = 0; b < 256; ++b) {
    running += hist[static_cast<std::size_t>(b)];
    cdf[static_cast<std::size_t>(b)] =
        static_cast<float>(running) / static_cast<float>( std::max(1, total_px));
  }

  for (int y = 0; y < lc.luma.height(); ++y) {
    for (int x = 0; x < lc.luma.width(); ++x) {
      const int bin = std::max(
          0, std::min(255, static_cast<int>(std::round(lc.luma.at(x, y, 0) * 255.0f))));
      lc.luma.at(x, y, 0) = cdf[static_cast<std::size_t>(bin)];
    }
  }

  return (src.channels() == 1) ? lc.luma : merge_luma_chroma(lc).clamped();
}

} // namespace imglib
