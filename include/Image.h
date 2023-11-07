#ifndef __IMAGE__H
#define __IMAGE__H

#include <cfloat>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "ImageException.h"


class Image {
public:
  // Constructor to initialize an image of size width*height*channels
  // Input Requirements:
  //      width   : a positive integer (>0) for the image width in pixels
  //      height  : positive integer for the image height in pixels
  //                 if not specified will default to 1
  //      channels: a value of either 1 or 3 for the image channels
  //                 if not specified will default to 1
  //      name    : string name for the image
  Image(int width, int height = 1, int channels = 1,
        const std::string &name = "");

  // Image Class Destructor. 
  ~Image();

  // Returns the image's name, should you specify one
  const std::string &name() const { return image_name; }

  // The distance between adjacent values in image_data in a given dimension
  // where width is dimension 0, height is dimension 1 and channel is
  // dimension 2. The dim input must be a value between 0 and 2
  int stride(int dim) const { return stride_[dim]; }

  int extent(int dim) const { return dim_values[dim]; } // Size of dimension
  int width() const { return dim_values[0]; }           // Extent of dimension 0
  int height() const { return dim_values[1]; }          // Extent of dimension 1
  int channels() const { return dim_values[2]; }        // Extent of dimension 2

  // set image pixels to corresponding values (only if channel is valid)
  void paint_color(float r = 0.0f, float g = 0.0f, float b = 0.0f);

  // set the rectangle bounded by [xstart, ystart] -> [xend, yend]
  // (inclusive) to specified color
  void draw_rectangle(int xstart, int ystart, int xend, int yend,
                        float r = 0.0f, float g = 0.0f, float b = 0.0f);

  // create a line segment from [xstart, ystart] to [xend, yend] with
  // specified color
  void draw_line(int xstart, int ystart, int xend, int yend, float r = 0.0f,
                   float g = 0.0f, float b = 0.0f);
  // The total number of elements.
  // Should be equal to width()*height()*channels()
  // That is, a 200x100x3 image has 60000 pixels not 20000 pixels
  long long len() const;

  // Getters for the pixel values
  const float &operator()(int x) const;
  const float &operator()(int x, int y) const;
  const float &operator()(int x, int y, int z) const;

  // Setters for the pixel values.
  // A reference to the value in image_data is returned
  float &operator()(int x);
  float &operator()(int x, int y);
  float &operator()(int x, int y, int z);

  // Safe Accessor that will return a black pixel (clamp = false) or the
  // nearest pixel value (clamp = true) when indexing out of the bounds of
  // the image
  float get(int x, int y, int z, bool clamp = false, float pad_value = 0.0) const;

  float min() const;
  float max() const;

  void copyFromGPU();
  void copyToGPU();
  void processWithCUDA();

  // The "private" section contains functions and variables that cannot be
  // accessed from outside the class.
private:
  static unsigned int const DIMS = 3; // Number of dimensions
  unsigned int dim_values[DIMS];      // Size of each dimension
  unsigned int stride_[DIMS];         // strides
  std::string image_name;             // Image name (filename if read from file)

  // This vector stores the values of the pixels. A vector in C++ is an array
  // that manages its own memory
  std::vector<float> image_data;

  // This does not allocate the image; it only initializes image metadata -
  // image name, width, height, number of channels and number of pixels
  void init_meta(int w, int h, int c, const std::string &name_);
  float *device_image_data;
};


void compareDimensions(const Image &im1, const Image &im2);

// Image/Image element-wise operations
Image operator+(const Image &im1, const Image &im2);
Image operator-(const Image &im1, const Image &im2);
Image operator*(const Image &im1, const Image &im2);
Image operator/(const Image &im1, const Image &im2);

// Image/scalar operations
Image operator+(const Image &im1, const float &c);
Image operator-(const Image &im1, const float &c);
Image operator*(const Image &im1, const float &c);
Image operator/(const Image &im1, const float &c);

// scalar/Image operations
Image operator+(const float &c, const Image &im1);
Image operator-(const float &c, const Image &im1);
Image operator*(const float &c, const Image &im1);
Image operator/(const float &c, const Image &im1);
// ------------------------------------------------------

#endif
