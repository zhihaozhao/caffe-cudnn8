/*
* @Author: Eric612
* @Date:   2018-08-20
* @https://github.com/eric612/Caffe-YOLOv2-Windows
* @https://github.com/eric612/MobileNet-YOLO
* Avisonic
*/
#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include "caffe/layers/yolov3_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include <algorithm>
#include <cfloat>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV


namespace caffe
{
#if 1
float biases[18] = {10, 13,  16, 30,  33, 23,  30, 61,  62, 45,  59, 119,  116, 90,  156, 198,  373, 326};
float biases_tiny[12] = {10, 14,  23, 27,  37, 58,  81, 82,  135, 169,  344, 319};

typedef struct
{
    float x, y, w, h;
} box;

typedef struct
{
    box bbox;
    int classes;
    float* prob;
    float objectness;
    int sort_class;
} detection;

typedef struct layer
{
    int batch;
    int total;
    int n, c, h, w;
    int out_n, out_c, out_h, out_w;
    int classes;
    int inputs, outputs;
    int mask[3];
    float* biases;
    float* output;

} layer;

//nms
int nms_comparator(const void* pa, const void* pb)
{
    detection a = *(detection*)pa;
    detection b = *(detection*)pb;
    float diff = 0;
    if (b.sort_class >= 0)
    {
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    }
    else
    {
        diff = a.objectness - b.objectness;
    }
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) return 0;
    float area = w * h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w * a.h + b.w * b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b) / box_union(a, b);
}

void do_nms_sort(detection* dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total - 1;
    for (i = 0; i <= k; ++i)
    {
        if (dets[i].objectness == 0)
        {
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (k = 0; k < classes; ++k)
    {
        for (i = 0; i < total; ++i)
        {
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator);
        for (i = 0; i < total; ++i)
        {
            if (dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for (j = i + 1; j < total; ++j)
            {
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh)
                {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

layer make_yolo_layer(int batch, int w, int h, int total, int classes, vector<int>& grid_size, float* output)
{
    layer l = {0};
    int n = 3;
    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n * (classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.inputs = l.w * l.h * l.c;

    if (total == 6)
    {
        l.biases = biases_tiny;

        if (w == grid_size[0])
        {
        	l.mask[0] = 3;
        	l.mask[1] = 4;
        	l.mask[2] = 5;
        }

        if (w == grid_size[1])
        {
        	l.mask[0] = 0;
        	l.mask[1] = 1;
        	l.mask[2] = 2;
        }
    }

    if (total == 9)
    {
        l.biases = biases;

        if (w == grid_size[0])
        {
        	l.mask[0] = 6;
        	l.mask[1] = 7;
        	l.mask[2] = 8;
        }

        if (w == grid_size[1])
        {
        	l.mask[0] = 3;
        	l.mask[1] = 4;
        	l.mask[2] = 5;
        }

        if (w == grid_size[2])
        {
        	l.mask[0] = 0;
        	l.mask[1] = 1;
        	l.mask[2] = 2;
        }
    }

    l.outputs = l.inputs;
    l.output = output;

    return l;
}


static int entry_index(layer l,int batch,int location,int entry)
{
    int n = location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4 + l.classes + 1) + entry*l.w*l.h + loc;
 }

static inline float logistic_activate(float x){return 1./(1. + exp(-x));}

void activate_array(float *x, const int n)
{
    int i;
    for(i = 0; i < n; ++i){
        x[i] = logistic_activate(x[i]);
    }
}

void forward_yolo_layer(layer l)
{
        for(int n = 0; n < l.n; ++n){
            int index = n*l.w*l.h*(4 + l.classes + 1);
            activate_array(l.output + index, 2*l.w*l.h);
            index = n*l.w*l.h*(4 + l.classes + 1) + 4*l.w*l.h;
            activate_array(l.output + index, (1+l.classes)*l.w*l.h);
        }

}

int yolo_num_detections(layer l,float thresh)
{
    int i,n,b;
    int count = 0;
  for(b = 0;b < l.batch;++b){
    for(i=0;i<l.w*l.h;++i){
        for(n=0;n<l.n;++n){
            int obj_index = entry_index(l,b,n*l.w*l.h+i,4);
            if(l.output[obj_index] > thresh)
                ++count;
        }	
    }
  }
  //printf("count = %d\n",count);
    return count;
}

int num_detections(vector<layer> layers_params,float thresh)
{
    int i;
    int s=0;
    for(i=0;i<layers_params.size();++i){
        layer l  = layers_params[i];
        s += yolo_num_detections(l,thresh);
    }
    return s;

}

detection* make_network_boxes(vector<layer> layers_params,float thresh,int* num)
{
    layer l = layers_params[0];
    int i;
    int nboxes = num_detections(layers_params,thresh);
    if(num) *num = nboxes;
    detection *dets = (detection*)calloc(nboxes,sizeof(detection));
    for(i=0;i<nboxes;++i){
        dets[i].prob = (float*)calloc(l.classes,sizeof(float));
    }
    return dets;
}


void correct_yolo_boxes(detection* dets,int n,int w,int h,int netw,int neth)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (w > h){
        new_w = netw;
        new_h = (h * netw)/w;
    }
    else{
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;

        dets[i].bbox = b;
    }
}


box get_yolo_box(float* x,float* biases,int n,int index,int i,int j,int lw, int lh,int w, int h,int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n] / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n + 1] / h;
    return b;
}


int get_yolo_detections(layer l, int netw,int neth,float thresh,detection *dets)
{
    int i,j,n,b;
    float* predictions = l.output;
    int count = 0;
  for(b = 0;b < l.batch;++b){
    for(i=0;i<l.w*l.h;++i){
        int row = i/l.w;
        int col = i%l.w;
        for(n = 0;n<l.n;++n){           
            int obj_index = entry_index(l,b,n*l.w*l.h + i,4);
            float objectness = predictions[obj_index];
            if(objectness <= thresh) continue;
            int box_index = entry_index(l,b,n*l.w*l.h + i,0);

            dets[count].bbox = get_yolo_box(predictions,l.biases,l.mask[n],box_index,col,row,l.w,l.h,netw,neth,l.w*l.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for(j=0;j<l.classes;++j){
                int class_index = entry_index(l,b,n*l.w*l.h+i,4+1+j);
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
  }
    
    return count;
}


void fill_network_boxes(vector<layer> layers_params, int netW, int netH, float thresh, detection *dets)
{
    int j;
    for(j=0;j<layers_params.size();++j){
        layer l = layers_params[j];
        int count = get_yolo_detections(l,netW,netH,thresh,dets);
        dets += count;
    }
}


detection* get_network_boxes(vector<layer> layers_params, int netW, int netH, float thresh,int *num)
{
    //make network boxes
    detection *dets = make_network_boxes(layers_params,thresh,num);

    //fill network boxes
    fill_network_boxes(layers_params,netW,netH,thresh,dets);
    return dets;
}

int get_output_boxes(detection *dets, int nboxes, int classes, float thresh)
{
	int i, j;
	int nbox = 0;

    for (i = 0; i < nboxes; ++i)
    {
        int cls = -1;
        for (j = 0; j < classes; ++j)
        {
            if (dets[i].prob[j] > thresh)
            {
                if (cls < 0)
                {
                    cls = j;
                }                
                break;
            }
        }
        if (cls >= 0)
        {
            nbox++;
        }
    }

    return nbox;
}

#endif
template <typename Dtype>
void Yolov3Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top)
{
    Yolov3Parameter yolov3_param = this->layer_param_.yolov3_param();

    classes_ = yolov3_param.classes();
    thresh_ = yolov3_param.thresh();

    net_w = bottom[0]->width() * 32;
  	net_h = bottom[0]->height() * 32;

  	maxBox =  0;

    for (int i = 0; i < bottom.size(); ++i)
    {
    	net_w = bottom[i]->width() * 32 < net_w ? bottom[i]->width() * 32 : net_w;
    	net_h = bottom[i]->height() * 32 < net_h ? bottom[i]->height() * 32 : net_h;

    	maxBox += bottom[i]->shape(2) * bottom[i]->shape(3) * 3;

    	grid_size.push_back(bottom[i]->shape(2));
    }

    std::sort(grid_size.begin(), grid_size.end());

    if (bottom.size() == 2)
    {
    	anchors_ = 6;
    }
    if (bottom.size() == 3)
    {
    	anchors_ = 9;
    }
}

template <typename Dtype>
void Yolov3Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top)
{
    // for (int i = 0; i < bottom.size(); ++i)
    // {
    //     LOG(INFO) << "bottom channels is  " << bottom[i]->channels();
    //     LOG(INFO) << "bottom width is  " << bottom[i]->height();
    //     LOG(INFO) << "bottom height is  " << bottom[i]->width();
    // }

    // num() and channels() are 1.
    vector<int> top_shape(2, 1);
    // set it to (fake) 90. (max output num)
    top_shape.push_back(90);
    // Each row is a 6 dimension vector, which stores
    // [label, confidence, x, y, w, h]
    top_shape.push_back(6);

    top[0]->Reshape(top_shape);

    yolo_output.Reshape(1, 1, maxBox, sizeof(detection)/sizeof(float) + classes_);
}


template <typename Dtype>
void Yolov3Layer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	int bottom_size = bottom.size();

    if (Caffe::mode() == Caffe::CPU)
    {
        for (int i = 0; i < bottom_size; ++i)
        {
            int w = bottom[i]->width();
            int h = bottom[i]->height();
            float* output = (float*)bottom[i]->mutable_cpu_data();

            for (int n = 0; n < 3; ++n)
            {
                int index = n*w*h*(4 + classes_ + 1);
                activate_array(output + index, 2 * w*h);
                index = n*w*h*(4 + classes_ + 1) + 4 * w*h;
                activate_array(output + index, (1 + classes_)*w*h);
            }
        }
    }

    vector<layer> layers_params;

    for(int i=0; i < bottom_size; ++i)
    {
    	int w = bottom[i]->width();
    	int h = bottom[i]->height();
    	Dtype* output = bottom[i]->mutable_cpu_data();
        layer l_params = make_yolo_layer(1,w,h,anchors_,classes_, grid_size, (float*)output);

        layers_params.push_back(l_params);               
    }    

    int nboxes = 0;

#if 0
  	detection* dets = get_network_boxes(layers_params,thresh_,&nboxes);
#else

  	nboxes = num_detections(layers_params, thresh_);
    float* data = (float*)yolo_output.mutable_cpu_data();
	detection *dets = (detection*)data;
    data += nboxes*sizeof(detection)/sizeof(float);

    for (int i = 0; i < nboxes; ++i)
    {
        dets[i].prob = data;
        data += classes_;
    }

    fill_network_boxes(layers_params, net_w, net_h, thresh_,dets);

#endif       
 
    do_nms_sort(dets,nboxes,classes_,0.45);	

	int out_boxes = get_output_boxes(dets, nboxes, classes_, thresh_);

	top[0]->Reshape(1, 1, out_boxes, 6);
	float* predict = (float*)top[0]->mutable_cpu_data();

	int obj = 0;

    int i, j;
    for (i = 0; i < nboxes; ++i)
    {
        int cls = -1;
        for (j = 0; j < classes_; ++j)
        {
            if (dets[i].prob[j] > thresh_)
            {
                if (cls < 0)
                {
                    cls = j;
                }                
                break;
            }
        }
        if (cls >= 0)
        {
            box b = dets[i].bbox;

            predict[obj*6 + 0] = cls;
			predict[obj*6 + 1] = dets[i].prob[cls];
			predict[obj*6 + 2] = b.x;
			predict[obj*6 + 3] = b.y;
			predict[obj*6 + 4] = b.w;
			predict[obj*6 + 5] = b.h;

			obj++;

			//printf("%d: %.0f%%\n", cls, dets[i].prob[cls] * 100);
            //printf("x = %f,y =  %f,w = %f,h =  %f\n",b.x,b.y,b.w,b.h);
        }
    }

}

#ifdef CPU_ONLY
STUB_GPU(Yolov3Layer);
#endif

INSTANTIATE_CLASS(Yolov3Layer);
REGISTER_LAYER_CLASS(Yolov3);

}  // namespace caffe
