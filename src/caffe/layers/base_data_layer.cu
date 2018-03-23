#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
void BasePrefetchingDataLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  // Note: this function runs in one thread per object and one object per one Solver thread
  shared_ptr<Batch> batch = this->batch_transformer_->processed_pop();
  if (batch_size_ == 1 || (last_shape_.size() > 0 && top[0]->shape() != last_shape_)) {
    top[0]->CopyDataFrom(*batch->data_, true);
  } else {
    top[0]->Swap(*batch->data_);
  }
  last_shape_ = top[0]->shape();
  if (this->output_labels_) {
      if (this->box_label_) {
          for(int i = 0; i < batch->multi_label_.size(); i++) { //TODO: prefetch count
            top[1]->Swap(*batch->multi_label_[i]);
          }
      } else {
          top[1]->Swap(*batch->label_);
      }
  }
  this->batch_transformer_->processed_push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD_ONLY_FB(BasePrefetchingDataLayer);

}  // namespace caffe
