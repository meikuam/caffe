#ifndef CAFFE_UTIL_GET_H_
#define CAFFE_UTIL_GET_H_

#include <cfloat>
#include <iostream>
#include "caffe/util/fp16_emu.h"
// #include "caffe/util/fp16_conversion.hpp"
#include "caffe/util/float16.hpp"

namespace caffe 
{
  template <typename T, typename Y>
  CAFFE_UTIL_IHD T Get(const Y& y)
  {
    return (T)y;
  }
  
  template <typename T>
  CAFFE_UTIL_IHD float tol(float t) {
    return t;
  }
  
  template <typename T>
  CAFFE_UTIL_IHD float maxDtype() {
    return FLT_MAX;
  }
  
  template <>
  CAFFE_UTIL_IHD float maxDtype<half>() {
    return HLF_MAX;
  }

  template <>
  CAFFE_UTIL_IHD float maxDtype<float16>() {
    return HLF_MAX;
  }

  template <typename T>
  CAFFE_UTIL_IHD float minDtype() {
    return FLT_MIN;
  }

  template <>
  CAFFE_UTIL_IHD float minDtype<half>() {
    return HLF_MIN;
  }

  template <>
  CAFFE_UTIL_IHD float minDtype<float16>() {
    return HLF_MIN;
  }

  template <>
  CAFFE_UTIL_IHD float tol<half>(float t) {
    return t < 1.e-4 ? 2.5e-2 : t * 2.5e2;
  }

}


#endif
