#include "../include/ImageManipulation.h"
using namespace std;

Image scaleNN(const Image &im, float factor) {
  // create a new image that is factor times bigger than the input by using
  // nearest neighbor interpolation.
  // return im;
  int ys, xs; // coordinate in the source image

  // Initialize a new Image factor times bigger (or smaller if factor <1)
  int nWidth = floor(factor * im.width());
  int nHeight = floor(factor * im.height());
  Image out(nWidth, nHeight, im.channels());

  // For each pixel in the output
  for (int z = 0; z < im.channels(); z++)
    for (int y = 0; y < nHeight; y++)
      for (int x = 0; x < nWidth; x++) {
        // Get the source pixel value.
        // If the output is factor times bigger, the source is 1/factor times
        // bigger...
        ys = round(1 / factor * y);
        xs = round(1 / factor * x);
        out(x, y, z) = im.get(xs, ys, z, true);
      }

  return out;
}

float interpolateLin(const Image &im, float x, float y, int z, bool clamp) {
  // bilinear interpolation samples the value of a non-integral
  // position (x,y) from its four "on-grid" neighboring pixels.
  //  |           |
  // -1-----------2-
  //  |           |  *: my coordinates (x,y) are not integral
  //  |  *        |     since I am not on the pixel grid :(
  //  |           |  1: top-left
  //  |           |  2: top-right
  //  |           |  3: bottom-right
  // -4-----------3- 4: bottom-left, what are our coordinates?
  //  |           |    We are willing to share some color
  //                   information with * ! Of course, the pixel
  //                   closest to * should influence it more.
  // return 0.0f;
  // get the neighboring points
  int xf = floor(x); // floor
  int yf = floor(y);
  int xc = xf + 1; // and ceil
  int yc = yf + 1;

  // compute the distances of the point to the floor-extreme point
  float yalpha = y - yf;
  float xalpha = x - xf;

  // obtain the values at those points
  float tl = im.get(xf, yf, z, clamp); // top-left
  float tr = im.get(xc, yf, z, clamp); // ...
  float bl = im.get(xf, yc, z, clamp);
  float br = im.get(xc, yc, z, clamp);

  // compute the interpolations on the top and bottom
  float topL = tr * xalpha + tl * (1.0f - xalpha);
  float botL = br * xalpha + bl * (1.0f - xalpha);

  // compute the overall interpolation
  float retv = botL * yalpha + topL * (1.0f - yalpha);

  // return final float value
  return retv;
}

Image scaleLin(const Image &im, float factor) {
  // create a new image that is factor times bigger than the input by using
  // bilinear interpolation
  // return im;
  float ys, xs; // coordinate in the source image

  // Initialize a new Image factor times bigger (or smaller if factor <1)
  int nWidth = floor(factor * im.width());
  int nHeight = floor(factor * im.height());
  Image im2(nWidth, nHeight, im.channels());

  // For each pixel in the output
  for (int z = 0; z < im.channels(); z++)
    for (int y = 0; y < nHeight; y++)
      for (int x = 0; x < nWidth; x++) {
        // Get the source pixel value.
        ys = 1 / factor * y;
        xs = 1 / factor * x;
        im2(x, y, z) = interpolateLin(im, xs, ys, z, true);
      }

  // return new image
  return im2;
}

Image scaleBicubic(const Image &im, float factor, float B, float C) {
  // create a new image that is factor times bigger than the input by using
  // a bicubic filter kernel with Mitchell and Netravali's parametrization
  // see "Reconstruction filters in computer graphics", Mitchell and Netravali
  // 1988 or http://entropymine.com/imageworsener/bicubic/

  // precompute coefficients
  float A3 = 2 - 1.5f * B - C;
  float A2 = -3 + 2 * B + C;
  float A0 = 1 - 0.33333f * B;

  float B3 = -0.166666f * B - C;
  float B2 = B + 5 * C;
  float B1 = -2 * B - 8 * C;
  float B0 = 1.333333f * B + 4 * C;

  // lambda function to compute the kernel weight
  auto computeK = [&](float x) -> float {
    float kx = 0.0f;
    float xabs = fabs(x);
    float x3 = pow(xabs, 3);
    float x2 = pow(xabs, 2);
    if (xabs < 1.0f) {
      kx = A3 * x3 + A2 * x2 + A0;
    } else if (1 <= xabs && xabs < 2.0f) {
      kx = B3 * x3 + B2 * x2 + B1 * xabs + B0;
    }
    return kx;
  };

  int nWidth = floor(factor * im.width());
  int nHeight = floor(factor * im.height());
  Image out(nWidth, nHeight, im.channels());

  for (int y = 0; y < nHeight; y++)
    for (int x = 0; x < nWidth; x++) {
      // Get the source pixel value.
      float ysrc = y / factor;
      float xsrc = x / factor;
      int xstart = (int)(floor(xsrc) - 2);
      int xend = (int)(floor(xsrc) + 2);
      int ystart = (int)(floor(ysrc) - 2);
      int yend = (int)(floor(ysrc) + 2);

      for (int ys = ystart; ys <= yend; ++ys)
        for (int xs = xstart; xs <= xend; ++xs) {
          float w = computeK(xsrc - xs) * computeK(ysrc - ys);
          for (int z = 0; z < im.channels(); z++)
            out(x, y, z) += im.get(xs, ys, z, false) * w;
        }
    }

  return out;
}

Image scaleLanczos(const Image &im, float factor, float a) {
  // create a new image that is factor times bigger than the input by using
  // a Lanczos filter kernel

  // lambda function to compute the kernel weight
  float PI2 = pow(M_PI, 2);
  float PI_A = M_PI / a;
  auto computeK = [&](float x) -> float {
    float kx = 1.0f;
    if (x != 0.0f && -a <= x && x < a) {
      kx = a * sin(M_PI * x) * sin(x * PI_A) / (PI2 * x * x);
    }
    return kx;
  };

  int nWidth = floor(factor * im.width());
  int nHeight = floor(factor * im.height());
  Image out(nWidth, nHeight, im.channels());

  for (int y = 0; y < nHeight; y++)
    for (int x = 0; x < nWidth; x++) {
      // Get the source pixel value.
      float ysrc = 1 / factor * y;
      float xsrc = 1 / factor * x;
      int xstart = (int)(floor(xsrc) - a + 1);
      int xend = (int)(floor(xsrc) + a);
      int ystart = (int)(floor(ysrc) - a + 1);
      int yend = (int)(floor(ysrc) + a);
      for (int xs = xstart; xs <= xend; ++xs)
        for (int ys = ystart; ys <= yend; ++ys) {
          float w = computeK(xsrc - xs) * computeK(ysrc - ys);
          for (int z = 0; z < im.channels(); z++)
            out(x, y, z) += im.get(xs, ys, z, false) * w;
        }
    }

  return out;
}

Image rotate(const Image &im, float theta) {
  // rotate an image around its center by theta

  // // center around which to rotate
  // float centerX = (im.width()-1.0)/2.0;
  // float centerY = (im.height()-1.0)/2.0;

  // center around which to rotate
  float centerX = (im.width() - 1.0) / 2.0;
  float centerY = (im.height() - 1.0) / 2.0;

  // get new image
  Image imR(im.width(), im.height(), im.channels());

  // For each pixel in the output
  float yR, xR; // rotated coordinates
  for (int x = 0; x < im.width(); x++)
    for (int y = 0; y < im.height(); y++)
      for (int z = 0; z < im.channels(); z++) {

        // compute the x and y values from the original image
        xR = (static_cast<float>(x) - centerX) * cos(theta) +
             (centerY - static_cast<float>(y)) * sin(theta) + centerX;
        yR = centerY - (-(static_cast<float>(x) - centerX) * sin(theta) +
                        (centerY - static_cast<float>(y)) * cos(theta));

        // interpolate the point
        imR(x, y, z) = interpolateLin(im, xR, yR, z);
      }

  return imR;
}


// Change the brightness of the image
Image brightness(const Image &im, float factor) {
  // Image output(im.width(), im.height(), im.channels());
  // Modify image brightness

  return im * factor;
}

Image contrast(const Image &im, float factor, float midpoint) {
  // Image output(im.width(), im.height(), im.channels());
  // Modify image contrast
  return (im - midpoint) * factor + midpoint;
}

Image color2gray(const Image &im, const std::vector<float> &weights) {
  // Image output(im.width(), im.height(), 1);
  // Convert to grayscale
  Image output(im.width(), im.height(), 1);
  for (int i = 0; i < im.width(); i++) {
    for (int j = 0; j < im.height(); j++) {
      float sum = 0;
      float weighted_sum = 0;
      for (int k = 0; k < im.channels(); k++) {
        weighted_sum += im(i, j, k) * weights[k];
        sum += weights[k];
      }
      output(i, j, 0) = weighted_sum / sum;
    }
  }
  return output;
}

// For this function, we want two outputs, a single channel luminance image
// and a three channel chrominance image. Return them in a vector with
// luminance first
std::vector<Image> lumiChromi(const Image &im) {
  // Create the luminance image
  // Create the chrominance image
  // Create the output vector as (luminance, chrominance)
  // Create the luminance
  Image im_luminance = color2gray(im);

  // Create chrominance images
  // We copy the input as starting point for the chrominance
  Image im_chrominance = im;
  for (int c = 0; c < im.channels(); c++)
    for (int y = 0; y < im.height(); y++)
      for (int x = 0; x < im.width(); x++) {
        im_chrominance(x, y, c) = im_chrominance(x, y, c) / im_luminance(x, y);
      }

  // Stack luminance and chrominance in the output vector, luminance first
  std::vector<Image> output;
  output.push_back(im_luminance);
  output.push_back(im_chrominance);
  return output;
}

Image lumiChromi2rgb(const vector<Image> &lc) {
  // luminance is lc[0]
  // chrominance is lc[1]

  // Create chrominance images
  // We copy the input as starting point for the chrominance
  Image im = Image(lc[1].width(), lc[1].height(), lc[1].channels());
  for (int c = 0; c < im.channels(); c++) {
    for (int y = 0; y < im.height(); y++) {
      for (int x = 0; x < im.width(); x++) {
        im(x, y, c) = lc[1](x, y, c) * lc[0](x, y);
      }
    }
  }
  return im;
}

// Modify brightness then contrast
Image brightnessContrastLumi(const Image &im, float brightF, float contrastF,
                             float midpoint) {
  // Modify brightness, then contrast of luminance image
  // Separate luminance and chrominance
  std::vector<Image> lumi_chromi = lumiChromi(im);
  Image im_luminance = lumi_chromi[0];
  Image im_chrominance = lumi_chromi[1];

  // Process the luminance channel
  im_luminance = brightness(im_luminance, brightF);
  im_luminance = contrast(im_luminance, contrastF, midpoint);

  // Multiply the chrominance with the new luminance to get the final image
  for (int i = 0; i < im.width(); i++) {
    for (int j = 0; j < im.height(); j++) {
      for (int c = 0; c < im.channels(); c++) {
        im_chrominance(i, j, c) = im_chrominance(i, j, c) * im_luminance(i, j);
      }
    }
  }
  // At this point, im_chrominance holds the complete processed image
  return im_chrominance;
}

Image rgb2yuv(const Image &im) {
  // Create output image of appropriate size
  // Change colorspace
  Image output(im.width(), im.height(), im.channels());
  for (int j = 0; j < im.height(); j++)
    for (int i = 0; i < im.width(); i++) {
      output(i, j, 0) =
          0.299 * im(i, j, 0) + 0.587 * im(i, j, 1) + 0.114 * im(i, j, 2);
      output(i, j, 1) =
          -0.147 * im(i, j, 0) - 0.289 * im(i, j, 1) + 0.436 * im(i, j, 2);
      output(i, j, 2) =
          0.615 * im(i, j, 0) - 0.515 * im(i, j, 1) - 0.100 * im(i, j, 2);
    }
  return output;
}

Image yuv2rgb(const Image &im) {
  // Create output image of appropriate size
  // Change colorspace
  Image output(im.width(), im.height(), im.channels());
  for (int j = 0; j < im.height(); j++)
    for (int i = 0; i < im.width(); i++) {
      output(i, j, 0) = im(i, j, 0) + 0 * im(i, j, 1) + 1.14 * im(i, j, 2);
      output(i, j, 1) = im(i, j, 0) - 0.395 * im(i, j, 1) - 0.581 * im(i, j, 2);
      output(i, j, 2) = im(i, j, 0) + 2.032 * im(i, j, 1) + 0 * im(i, j, 2);
    }
  return output;
}

Image saturate(const Image &im, float factor) {
  // Create output image of appropriate size
  // Saturate image
  Image output = rgb2yuv(im); // Change colorspace
  for (int i = 0; i < im.width(); i++) {
    for (int j = 0; j < im.height(); j++) {
      output(i, j, 1) = output(i, j, 1) * factor;
      output(i, j, 2) = output(i, j, 2) * factor;
    }
  }
  output = yuv2rgb(output); // Back to RGB
  return output;
}

// Gamma codes the image
Image gamma_code(const Image &im, float gamma) {
  // Image output(im.width(), im.height(), im.channels());
  // Gamma encodes the image
  Image output = Image(im.width(), im.height(), im.channels());
  for (int i = 0; i < im.len(); ++i) {
    output(i) = pow(im(i), (1 / gamma));
  }
  return output;
}

// Quantizes the image to 2^bits levels and scales back to 0~1
Image quantize(const Image &im, int bits) {
  // Image output(im.width(), im.height(), im.channels());
  // Quantizes the image to 2^bits levels
  Image output(im.width(), im.height(), im.channels());
  for (int i = 0; i < im.width(); i++) {
    for (int j = 0; j < im.height(); j++) {
      for (int k = 0; k < im.channels(); k++) {
        output(i, j, k) = roundf(im(i, j, k) * (pow(2.f, (float)bits) - 1)) /
                          (pow(2.f, (float)bits) - 1.f);
      }
    }
  }
  return output;
}

// Compare between first quantize then gamma_encode and first gamma_encode
// then quantize
std::vector<Image> gamma_test(const Image &im, int bits, float gamma) {
  Image im1 = quantize(im, bits);
  im1 = gamma_code(im1, gamma);

  Image im2 = gamma_code(im, gamma);
  im2 = quantize(im2, bits);

  std::vector<Image> imgs;
  imgs.push_back(im1);
  imgs.push_back(im2);
  return imgs;
}

// Return two images in a C++ vector
std::vector<Image> spanish(const Image &im) {
  // Extract the luminance
  Image output_L = color2gray(im);

  // Convert to YUV for manipulation
  Image output_C = rgb2yuv(im);

  for (int j = 0; j < im.height(); j++)
    for (int i = 0; i < im.width(); i++) {
      output_C(i, j, 0) = 0.5;                // constant luminance
      output_C(i, j, 1) = -output_C(i, j, 1); // opposite chrominance
      output_C(i, j, 2) = -output_C(i, j, 2); // opposite chrominance
    }
  // Convert back to RGB
  output_C = yuv2rgb(output_C);

  // Location of the black dot
  int bdot_x = floor(im.width() / 2);
  int bdot_y = floor(im.height() / 2);

  // Add the black dot to Luminance, and Chrominance images
  output_L(bdot_x, bdot_y, 0) = 0.0f;
  output_C(bdot_x, bdot_y, 0) = 0.0f; // black is 0
  output_C(bdot_x, bdot_y, 1) = 0.0f;
  output_C(bdot_x, bdot_y, 2) = 0.0f;

  // Pack the images in a vector, chrominance first
  std::vector<Image> output;
  output.push_back(output_C);
  output.push_back(output_L);
  return output;
}

// White balances an image using the gray world assumption
Image grayworld(const Image &im) {
  // Compute the mean per channel
  // find green mean
  float green_sum = 0;
  Image output(im.width(), im.height(), im.channels());
  for (int y = 0; y < im.height(); y++) {
    for (int x = 0; x < im.width(); x++) {
      green_sum += im(x, y, 1);
    }
  }
  float green_mean = green_sum / (im.width() * im.height());

  // red
  float red_sum = 0;
  for (int y = 0; y < im.height(); y++) {
    for (int x = 0; x < im.width(); x++) {
      red_sum += im(x, y, 0);
    }
  }
  float red_mean = red_sum / (im.width() * im.height());

  // blue
  float blue_sum = 0;
  for (int y = 0; y < im.height(); y++) {
    for (int x = 0; x < im.width(); x++) {
      blue_sum += im(x, y, 2);
    }
  }
  float blue_mean = blue_sum / (im.width() * im.height());

  float red_factor = green_mean / red_mean;
  float blue_factor = green_mean / blue_mean;

  // normalize 'white balance' red & blue
  for (int y = 0; y < im.height(); y++) {
    for (int x = 0; x < im.width(); x++) {
      output(x, y, 0) = im(x, y, 0) * (red_factor);
      output(x, y, 1) = im(x, y, 1);
      output(x, y, 2) = im(x, y, 2) * (blue_factor);
    }
  }

  return output;
}