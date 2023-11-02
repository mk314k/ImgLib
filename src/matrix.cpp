// #include "../include/matrix.h"

// using namespace std;


// // obtain minimum pixel value
// float Matrix::min() const {
//   float minf = FLT_MAX;
//   for (int i = 0; i < len(); i++) {
//     minf = std::min(minf, (*this)(i));
//   }
//   return minf;
// }

// // obtain maximum pixel value
// float Matrix::max() const {
//   float maxf = -FLT_MAX;
//   for (int i = 0; i < len(); i++) {
//     maxf = std::max(maxf, (*this)(i));
//   }
//   return maxf;
// }


// long long Matrix::len() const {
//   // returns the number of elements in the image.
//   // An RGB (3 color channels) image of 100 × 100 pixels has 30000 elements
//   return data.size();
// }

// // -------------- Accessors and Setters -----------------
// const float &Matrix::operator()(int x) const {
//   // Linear accessor to the image data
//   if (x < 0 || x >= len())
//     throw OutOfBoundsException();
//   return data[x];
// }

// const float &Matrix::operator()(int x, int y) const {
//   // Accessor to the image data at channel 0
//   if (x < 0 || x >= width())
//     throw OutOfBoundsException();
//   if (y < 0 || y >= height())
//     throw OutOfBoundsException();

//   return data[x * stride_[0] + y * stride_[1]];
// }


// float &Matrix::operator()(int x) {
//   // Linear setter to the image data
//   if (x < 0 || x >= len())
//     throw OutOfBoundsException();
//   return data[x];
// }

// float &Matrix::operator()(int x, int y) {
//   // Setter to the image data at channel 0
//   if (x < 0 || x >= width())
//     throw OutOfBoundsException();
//   if (y < 0 || y >= height())
//     throw OutOfBoundsException();

//   return data[x * stride_[0] + y * stride_[1]];
// }



// Matrix::Matrix(int width, int height) {
//   if (width < 1)
//     throw IllegalDimensionException();
//   if (height < 1)
//     throw IllegalDimensionException();

//   // Initialize image data
//   dim_values[0] = width;
//   dim_values[1] = height;
//   stride_[0] = 1;
//   stride_[1] = width;

//   long long size_of_data = 1;
//   for (unsigned int k = 0; k < DIMS; k++) {
//     size_of_data *= dim_values[k];
//   }
//   data = std::vector<float>(size_of_data, 0.f);
// }
// Matrix::~Matrix(){};

// void compareDimensions(const Matrix &im1, const Matrix &im2) {
//   for (int i = 0; i < 3; i++) {
//     if (im1.extent(i) != im2.extent(i))
//       throw MismatchedDimensionsException();
//   }
// }

// Matrix operator+(const Matrix &im1, const Matrix &im2) {
//   compareDimensions(im1, im2);
//   long long total_pixels = im1.len();

//   Matrix output(im1.extent(0), im1.extent(1));
//   for (int i = 0; i < total_pixels; i++) {
//     output(i) = im1(i) + im2(i);
//   }
//   return output;
// }

// Matrix operator-(const Matrix &im1, const Matrix &im2) {
//   compareDimensions(im1, im2);
//   long long total_pixels = im1.len();
//   Matrix output(im1.extent(0), im1.extent(1));
//   for (int i = 0; i < total_pixels; i++) {
//     output(i) = im1(i) - im2(i);
//   }
//   return output;
// }

// Matrix operator*(const Matrix &im1, const Matrix &im2) {
//   compareDimensions(im1, im2);
//   long long total_pixels = im1.len();
//   Matrix output(im1.extent(0), im1.extent(1));
//   for (int i = 0; i < total_pixels; i++) {
//     output(i) = im1(i) * im2(i);
//   }
//   return output;
// }

// Matrix operator/(const Matrix &im1, const Matrix &im2) {
//   compareDimensions(im1, im2);
//   long long total_pixels = im1.len();
//   Matrix output(im1.extent(0), im1.extent(1));
//   for (int i = 0; i < total_pixels; i++) {
//     if (im2(i) == 0)
//       throw DivideByZeroException();
//     output(i) = im1(i) / im2(i);
//   }
//   return output;
// }

// Matrix operator+(const Matrix &im1, const float &c) {
//   long long total_pixels = im1.len();
//   Matrix output(im1.extent(0), im1.extent(1));
//   for (int i = 0; i < total_pixels; i++) {
//     output(i) = im1(i) + c;
//   }
//   return output;
// }

// Matrix operator-(const Matrix &im1, const float &c) {
//   long long total_pixels = im1.len();
//   Matrix output(im1.extent(0), im1.extent(1));
//   for (int i = 0; i < total_pixels; i++) {
//     output(i) = im1(i) - c;
//   }
//   return output;
// }
// Matrix operator*(const Matrix &im1, const float &c) {
//   long long total_pixels = im1.len();
//   Matrix output(im1.extent(0), im1.extent(1));
//   for (int i = 0; i < total_pixels; i++) {
//     output(i) = im1(i) * c;
//   }
//   return output;
// }
// Matrix operator/(const Matrix &im1, const float &c) {
//   long long total_pixels = im1.len();
//   Matrix output(im1.extent(0), im1.extent(1));
//   if (c == 0)
//     throw DivideByZeroException();
//   for (int i = 0; i < total_pixels; i++) {
//     output(i) = im1(i) / c;
//   }
//   return output;
// }

// Matrix operator+(const float &c, const Matrix &im1) {
//   long long total_pixels = im1.len();
//   Matrix output(im1.extent(0), im1.extent(1));
//   for (int i = 0; i < total_pixels; i++) {
//     output(i) = im1(i) + c;
//   }
//   return output;
// }

// Matrix operator-(const float &c, const Matrix &im1) {
//   long long total_pixels = im1.len();
//   Matrix output(im1.extent(0), im1.extent(1));
//   for (int i = 0; i < total_pixels; i++) {
//     output(i) = c - im1(i);
//   }
//   return output;
// }

// Matrix operator*(const float &c, const Matrix &im1) {
//   long long total_pixels = im1.len();
//   Matrix output(im1.extent(0), im1.extent(1));
//   for (int i = 0; i < total_pixels; i++) {
//     output(i) = im1(i) * c;
//   }
//   return output;
// }
// Matrix operator/(const float &c, const Matrix &im1) {
//   long long total_pixels = im1.len();
//   Matrix output(im1.extent(0), im1.extent(1));
//   for (int i = 0; i < total_pixels; i++) {
//     if (im1(i) == 0)
//       throw DivideByZeroException();
//     output(i) = c / im1(i);
//   }
//   return output;
// }
