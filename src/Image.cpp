#include "../include/Image.h"

using namespace std;


// obtain minimum pixel value
float Image::min() const {
  float minf = FLT_MAX;
  for (int i = 0; i < len(); i++) {
    minf = std::min(minf, (*this)(i));
  }
  return minf;
}

// obtain maximum pixel value
float Image::max() const {
  float maxf = -FLT_MAX;
  for (int i = 0; i < len(); i++) {
    maxf = std::max(maxf, (*this)(i));
  }
  return maxf;
}

// Safe Accessor that will return a black pixel (clamp = false) or the nearest
// pixel value (clamp = true) when indexing out of the bounds of the image
float Image::get(int x, int y, int z,bool clamp, float pad_value) const {
  int x0 = x;
  int y0 = y;

  if (clamp){
    // clamp to nearest boundary in case out of bounds
    if (y>=height()){
      y0= height() -1;
    }
    if (y<0){
      y0 = 0;
    }
    if (x>=width()){
      x0 = width() -1;
    }
    if (x<0){
      x0=0;
    }
  }else{
    //out of boundary means black padding
    if (
      y>=height() ||
      y<0 ||
      x>=width() ||
      x<0
    ){
      return pad_value;
    }
  }

  return (*this)(x0, y0, z);
}

long long Image::len() const {
  // returns the number of elements in the image.
  // An RGB (3 color channels) image of 100 × 100 pixels has 30000 elements
  return image_data.size();
}

// -------------- Accessors and Setters -----------------
const float &Image::operator()(int x) const {
  // Linear accessor to the image data
  if (x < 0 || x >= len())
    throw OutOfBoundsException();
  return image_data[x];
}

const float &Image::operator()(int x, int y) const {
  // Accessor to the image data at channel 0
  if (x < 0 || x >= width())
    throw OutOfBoundsException();
  if (y < 0 || y >= height())
    throw OutOfBoundsException();

  return image_data[x * stride_[0] + y * stride_[1]];
}

const float &Image::operator()(int x, int y, int z) const {
  // Accessor to the image data at channel z
  if (x < 0 || x >= width())
    throw OutOfBoundsException();
  if (y < 0 || y >= height())
    throw OutOfBoundsException();
  if (z < 0 || z >= channels())
    throw OutOfBoundsException();

  return image_data[x * stride_[0] + y * stride_[1] + stride_[2] * z];
}

float &Image::operator()(int x) {
  // Linear setter to the image data
  if (x < 0 || x >= len())
    throw OutOfBoundsException();
  return image_data[x];
}

float &Image::operator()(int x, int y) {
  // Setter to the image data at channel 0
  if (x < 0 || x >= width())
    throw OutOfBoundsException();
  if (y < 0 || y >= height())
    throw OutOfBoundsException();

  return image_data[x * stride_[0] + y * stride_[1]];
}

float &Image::operator()(int x, int y, int z) {
  // Setter to the image data at channel z
  if (x < 0 || x >= width())
    throw OutOfBoundsException();
  if (y < 0 || y >= height())
    throw OutOfBoundsException();
  if (z < 0 || z >= channels())
    throw OutOfBoundsException();

  return image_data[x * stride_[0] + y * stride_[1] + stride_[2] * z];
}

void Image::paint_color(float r, float g, float b) {
  // Set the image pixels to the corresponding values
  for (int i = 0; i < width() * height(); ++i) {
    image_data[i] = r;
    if (channels() > 1) // have second channel
      image_data[i + stride_[2]] = g;
    if (channels() > 2) // have third channel
      image_data[i + 2 * stride_[2]] = b;
  }
}

void Image::draw_rectangle(int xstart, int ystart, int xend, int yend,
                             float r, float g, float b) {
  // Set the pixels inside the rectangle to the specified color

  if (xstart < 0 || xstart >= width() || ystart < 0 || ystart >= height())
    throw OutOfBoundsException();
  if (xend < 0 || xend >= width() || yend < 0 || yend >= height())
    throw OutOfBoundsException();

  float col[3] = {r, g, b};
  for (int w = xstart; w <= xend; ++w) {
    for (int h = ystart; h <= yend; ++h) {
      for (int c = 0; c < channels(); ++c) {
        (*this)(w, h, c) = col[c];
      }
    }
  }
}

void Image::draw_line(int xstart, int ystart, int xend, int yend, float r,
                        float g, float b) {
  // Create a line segment with specified color

  if (xstart < 0 || xstart >= width() || ystart < 0 || ystart >= height())
    throw OutOfBoundsException();
  if (xend < 0 || xend >= width() || yend < 0 || yend >= height())
    throw OutOfBoundsException();

  int valid_channels = channels() > 3 ? 3 : channels();
  float col[3] = {r, g, b};
  float x = xstart;
  float y = ystart;
  int delta_x = xend - xstart;
  int delta_y = yend - ystart;
  int delta = std::max(std::abs(delta_x), std::abs(delta_y));
  for (int i = 0; i <= delta; i++) {
    int ix = std::round(x), iy = std::round(y);
    if (ix >= 0 && ix < width() && iy >= 0 && iy < height()) {
      for (int c = 0; c < valid_channels; ++c) {
        (*this)(ix, iy, c) = col[c];
      }
    }
    x += (float(delta_x) / float(delta));
    y += (float(delta_y) / float(delta));
  }
}


Image::Image(int width, int height, int channels ,const std::string &name_) {
  init_meta(width, height, channels, name_);
  // Initialize image data
  long long size_of_data = 1;
  for (unsigned int k = 0; k < DIMS; k++) {
    size_of_data *= dim_values[k];
  }
  image_data = std::vector<float>(size_of_data, 0.f);
}

void Image::init_meta(int w, int h, int c,
                                      const std::string &name_) {
  if (w < 1)
    throw IllegalDimensionException();
  if (h < 1)
    throw IllegalDimensionException();
  if (c != 1 && c != 3)
    throw IllegalDimensionException();

  dim_values[0] = w;
  dim_values[1] = h;
  dim_values[2] = c;
  stride_[0] = 1;
  stride_[1] = w;
  stride_[2] = w * h;
  image_name = name_;
}
void compareDimensions(const Image &im1, const Image &im2) {
  for (int i = 0; i < 3; i++) {
    if (im1.extent(i) != im2.extent(i))
      throw MismatchedDimensionsException();
  }
}

Image operator+(const Image &im1, const Image &im2) {
  compareDimensions(im1, im2);
  long long total_pixels = im1.len();

  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  for (int i = 0; i < total_pixels; i++) {
    output(i) = im1(i) + im2(i);
  }
  return output;
}

Image operator-(const Image &im1, const Image &im2) {
  compareDimensions(im1, im2);
  long long total_pixels = im1.len();
  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  for (int i = 0; i < total_pixels; i++) {
    output(i) = im1(i) - im2(i);
  }
  return output;
}

Image operator*(const Image &im1, const Image &im2) {
  compareDimensions(im1, im2);
  long long total_pixels = im1.len();
  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  for (int i = 0; i < total_pixels; i++) {
    output(i) = im1(i) * im2(i);
  }
  return output;
}

Image operator/(const Image &im1, const Image &im2) {
  compareDimensions(im1, im2);
  long long total_pixels = im1.len();
  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  for (int i = 0; i < total_pixels; i++) {
    if (im2(i) == 0)
      throw DivideByZeroException();
    output(i) = im1(i) / im2(i);
  }
  return output;
}

Image operator+(const Image &im1, const float &c) {
  long long total_pixels = im1.len();
  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  for (int i = 0; i < total_pixels; i++) {
    output(i) = im1(i) + c;
  }
  return output;
}

Image operator-(const Image &im1, const float &c) {
  long long total_pixels = im1.len();
  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  for (int i = 0; i < total_pixels; i++) {
    output(i) = im1(i) - c;
  }
  return output;
}
Image operator*(const Image &im1, const float &c) {
  long long total_pixels = im1.len();
  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  for (int i = 0; i < total_pixels; i++) {
    output(i) = im1(i) * c;
  }
  return output;
}
Image operator/(const Image &im1, const float &c) {
  long long total_pixels = im1.len();
  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  if (c == 0)
    throw DivideByZeroException();
  for (int i = 0; i < total_pixels; i++) {
    output(i) = im1(i) / c;
  }
  return output;
}

Image operator+(const float &c, const Image &im1) {
  long long total_pixels = im1.len();
  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  for (int i = 0; i < total_pixels; i++) {
    output(i) = im1(i) + c;
  }
  return output;
}

Image operator-(const float &c, const Image &im1) {
  long long total_pixels = im1.len();
  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  for (int i = 0; i < total_pixels; i++) {
    output(i) = c - im1(i);
  }
  return output;
}

Image operator*(const float &c, const Image &im1) {
  long long total_pixels = im1.len();
  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  for (int i = 0; i < total_pixels; i++) {
    output(i) = im1(i) * c;
  }
  return output;
}
Image operator/(const float &c, const Image &im1) {
  long long total_pixels = im1.len();
  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  for (int i = 0; i < total_pixels; i++) {
    if (im1(i) == 0)
      throw DivideByZeroException();
    output(i) = c / im1(i);
  }
  return output;
}
