#ifndef CAFFE_YOLOV3_LAYER_HPP_
#define CAFFE_YOLOV3_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include <string>

namespace caffe
{
template <typename Dtype>
class Yolov3Layer : public Layer<Dtype>
{
public:
    explicit Yolov3Layer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "Yolov3"; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
  	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  	}

 	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
							 const vector<Blob<Dtype>*>& top);

 	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  	}

    int classes_;
  	float thresh_;
  	int anchors_;
  	int maxBox;
  	int net_w;
  	int net_h;
  	Blob<Dtype> yolo_output;
  	vector<int> grid_size;
};

}  // namespace caffe

#endif  // CAFFE_YOLOV3_LAYER_HPP_
