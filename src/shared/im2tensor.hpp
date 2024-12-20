#include "atoms/image_path_vo.hpp"
#include "atoms/tensor_vo.hpp"

/*
Image Processing: Load the image and put it into a tensor.
    image: ImagePathVo
    tensorImage: TensorVo
*/
void im2tensor(TensorVo tensorImage, ImagePathVo image); // TODO passer sous GPU. peut-être