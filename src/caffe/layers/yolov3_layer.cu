// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <vector>

#include "caffe/layers/yolov3_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SigmoidForward(const int n, Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = 1./(1. + exp(-in[index]));    
  }
}

template <typename Dtype>
void Yolov3Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

#if 1
  for(int i = 0; i <  bottom.size(); i++)
  {
    Dtype* bottom_data = bottom[i]->mutable_gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int count = bottom[i]->count();

      int w = bottom[i]->width();
      int h = bottom[i]->height();

      for (int n = 0; n < 3; ++n)
      {
        int index = n*w*h*(4 + classes_ + 1);
        int num = 2*w*h;

        SigmoidForward<Dtype><<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS>>>(
        num, bottom_data + index, bottom_data + index);

        num = (1+classes_)*w*h;  
        index = n*w*h*(4 + classes_ + 1) + 4*w*h;

        SigmoidForward<Dtype><<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS>>>(
        num, bottom_data + index, bottom_data + index);    
      }

    CUDA_POST_KERNEL_CHECK;
  }

#endif

  Forward_cpu(bottom, top);
}

INSTANTIATE_LAYER_GPU_FUNCS(Yolov3Layer);

}  // namespace caffe
