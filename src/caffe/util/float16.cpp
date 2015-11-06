// Utility contants
#include "caffe/util/float16.hpp"
#include <iostream>

#if !NEW_COOL_FP16
namespace caffe 
{
  const float16 float16::zero = float16(0.); 
  const float16 float16::one = float16(1.); 
  const float16 float16::minus_one = float16(-1.); 


  std::ostream& operator << (std::ostream& s, const float16& f) 
  {
    return s<<(float)f;
  }

}
#endif
