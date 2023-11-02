// /* --------------------------------------------------------------------------
//  * File:    matrix.h
//  * Author:  Michael Gharbi <gharbi@mit.edu>
//  * Created: 2015-10-17
//  * --------------------------------------------------------------------------
//  *
//  *
//  *
//  * ------------------------------------------------------------------------*/

// // #pragma once

// // #include <./Eigen/Dense>
// // // using namespace Eigen;

// // typedef Eigen::MatrixXf Matrix;
// // typedef Eigen::Vector2f Vec2f;
// // typedef Eigen::Vector3f Vec3f;

// #ifndef __MATRIX__H
// #define __MATRIX__H

// #include <cfloat>
// #include <cmath>
// #include <iostream>
// #include <string>
// #include <vector>

// #include "ImageException.h"


// class Matrix {
// public:
//   // Constructor to initialize an image of size width*height*channels
//   // Input Requirements:
//   //      width   : a positive integer (>0) for the image width in pixels
//   //      height  : positive integer for the image height in pixels
//   //                 if not specified will default to 1
//   //      channels: a value of either 1 or 3 for the image channels
//   //                 if not specified will default to 1
//   //      name    : string name for the image
//   Matrix(int width, int height);

//   // Image Class Destructor. 
//   ~Matrix();


//   // The distance between adjacent values in image_data in a given dimension
//   // where width is dimension 0, height is dimension 1 and channel is
//   // dimension 2. The dim input must be a value between 0 and 2
//   int stride(int dim) const { return stride_[dim]; }

//   int extent(int dim) const { return dim_values[dim]; } // Size of dimension
//   int width() const { return dim_values[0]; }           // Extent of dimension 0
//   int height() const { return dim_values[1]; }          // Extent of dimension 1

//   // The total number of elements.
//   // Should be equal to width()*height()*channels()
//   // That is, a 200x100x3 image has 60000 pixels not 20000 pixels
//   long long len() const;

//   // Getters for the pixel values
//   const float &operator()(int x) const;
//   const float &operator()(int x, int y) const;

//   // Setters for the pixel values.
//   // A reference to the value in image_data is returned
//   float &operator()(int x);
//   float &operator()(int x, int y);


//   float min() const;
//   float max() const;

//   // The "private" section contains functions and variables that cannot be
//   // accessed from outside the class.
// private:
//   static unsigned int const DIMS = 2; // Number of dimensions
//   unsigned int dim_values[DIMS];      // Size of each dimension
//   unsigned int stride_[DIMS];         // strides

//   // This vector stores the values of the pixels. A vector in C++ is an array
//   // that manages its own memory
//   std::vector<float> data;
// };


// void compareDimensions(const Matrix &im1, const Matrix &im2);

// // Image/Image element-wise operations
// Matrix operator+(const Matrix &im1, const Matrix &im2);
// Matrix operator-(const Matrix &im1, const Matrix &im2);
// Matrix operator*(const Matrix &im1, const Matrix &im2);
// Matrix operator/(const Matrix &im1, const Matrix &im2);

// // Matrix/scalar operations
// Matrix operator+(const Matrix &im1, const float &c);
// Matrix operator-(const Matrix &im1, const float &c);
// Matrix operator*(const Matrix &im1, const float &c);
// Matrix operator/(const Matrix &im1, const float &c);

// // scalar/Matrix operations
// Matrix operator+(const float &c, const Matrix &im1);
// Matrix operator-(const float &c, const Matrix &im1);
// Matrix operator*(const float &c, const Matrix &im1);
// Matrix operator/(const float &c, const Matrix &im1);
// // ------------------------------------------------------

// #endif
