#include "atoms/image_path_vo.hpp"
#include "atoms/tensor_vo.hpp"
#include <stdint.h>

/*
transpose: Transpose a tensor depth-wise
    tensor: TensorVo
    Return tensor.T
*/ 
TensorVo transpose(MatrixVo tensor, unsigned int width, unsigned int height, uint8_t depth);