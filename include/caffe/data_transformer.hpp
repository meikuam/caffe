#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/io.hpp"

#include "google/protobuf/repeated_field.h"
using google::protobuf::RepeatedPtrField;

namespace caffe {

class BoxLabel {
 public:
  float class_label_;
  float difficult_;
  float box_[4];
};

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, subtracting the image mean...
 */
template<typename Dtype>
class DataTransformer {
 public:
  DataTransformer(const TransformationParameter& param, Phase phase);
  ~DataTransformer() = default;

  const TransformationParameter& transform_param() const {
    return param_;
  }

  /**
   * @brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  void InitRand();

  void TransformGPU(int N, int C, int H, int W, size_t sizeof_element,
      const void* in, Dtype* out, const unsigned int* rands, bool signed_data);

  /**
   * @brief Applies transformations defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum [in]
   *    The source Datum containing data of arbitrary shape.
   * @param buf_len [in]
   *    Buffer length in Dtype elements
   * @param buf [out]
   *    The destination array that will store transformed data of a fixed
   *    shape. If nullptr passed then only shape vector is computed.
   * @return Output shape
   */
  vector<int> Transform(const Datum* datum, Dtype* buf, size_t buf_len,
      Packing& out_packing, bool repack = true);

  /**
   * @brief Applies transformations defined in the image data layer's
   * transform_param block to the data.
   *
   * @param datum [in]
   *    The source cv::Mat containing data of arbitrary shape.
   * @param buf_len [in]
   *    Buffer length in Dtype elements
   * @param buf [out]
   *    The destination array that will store transformed data of a fixed
   *    shape.
   */
  void Transform(const cv::Mat& src, Dtype* buf, size_t buf_len, bool repack = true) const;

  void Transform(const cv::Mat& img, TBlob<Dtype> *transformed_blob) const;

    /**
     * @brief Applies the transformation defined in the data layer's
     * transform_param block to a vector of Mat.
     *
     * @param mat_vector
     *    A vector of Mat containing the data to be transformed.
     * @param transformed_blob
     *    This is destination blob. It can be part of top blob's data if
     *    set_cpu_data() is used. See memory_layer.cpp for an example.
     */
  void Transform(const vector<cv::Mat>& mat_vector, TBlob<Dtype>* transformed_blob) const;

  void Transform(const vector<Datum>& datum_vector, TBlob<Dtype> *transformed_blob) const;

  /**
   * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
   *
   * @param n
   *    The upperbound (exclusive) value of the random number.
   * @return
   *    A uniformly random integer value from ({0, 1, ..., n-1}).
   */
  unsigned int Rand(int n) const {
    CHECK_GT(n, 0);
    return Rand() % n;
  }
  void Transform(const Datum& datum, TBlob<Dtype>* transformed_blob, vector<BoxLabel>* box_labels) {
      int float_size = datum.float_data_size();
        CHECK_GT(float_size, 0) <<
          "Every sample must have label";
        CHECK_EQ(float_size % 6, 0) <<
          "Every box label has 6 labels (class, difficult, box)";
        //*********************************************************************//
        //get ori_labels
        vector<BoxLabel> ori_labels;
        for (int j = 0; j < float_size; j += 6) {
          BoxLabel box_label;
          box_label.class_label_ = datum.float_data(j);
          box_label.difficult_ = datum.float_data(j + 1);
          for (int k = 2; k < 6; ++k) {
            box_label.box_[k-2] = datum.float_data(j+k);
          }
          ori_labels.push_back(box_label);
        }

        // If datum is encoded, decoded and transform the cv::image.
        CHECK(datum.encoded()) << "For box data, datum must be encoded";
        CHECK(!(param_.force_color() && param_.force_gray()))
          << "cannot set both force_color and force_gray";
        cv::Mat cv_img;
        if (param_.force_color() || param_.force_gray()) {
        // If force_color then decode in color otherwise decode in gray.
          cv_img = DecodeDatumToCVMat(datum, param_.force_color());
        } else {
          cv_img = DecodeDatumToCVMatNative(datum);
        }

        if (phase_ == TEST) {
          *box_labels = ori_labels;
          Transform(cv_img, transformed_blob);
          return;
        }

        int img_width = cv_img.cols;
        int img_height = cv_img.rows;

        cv::Mat cv_rand_img;
        // bool mirror = Rand(2);
        bool mirror = 0;
        while (box_labels->size() == 0) {
          float rand_scale = (1. - Rand(30) / 100.);
          int rand_w = static_cast<int>(img_width * rand_scale) - 1;
          int rand_h = static_cast<int>(img_height * rand_scale) - 1;
          // LOG(INFO) << "rand_w: " << rand_w << " rand_h: " << rand_h;
          // LOG(INFO) << "img_width: " << img_width << " img_height: " << img_height;
          int rand_x = Rand(img_width - rand_w);
          int rand_y = Rand(img_height - rand_h);
          for (int i = 0; i < ori_labels.size(); ++i) {
            int ori_x = static_cast<int>(ori_labels[i].box_[0] * img_width); //box center(x,y)
            int ori_y = static_cast<int>(ori_labels[i].box_[1] * img_height);
            int ori_w = static_cast<int>(ori_labels[i].box_[2] * img_width); //box(w,h)
            int ori_h = static_cast<int>(ori_labels[i].box_[3] * img_height);
            if (!(ori_x >= rand_x && ori_x < rand_x + rand_w)) {
              continue;
            }
            if (!(ori_y >= rand_y && ori_y < rand_y + rand_h)) {
              continue;
            }
            BoxLabel box_label;
            box_label.difficult_ = ori_labels[i].difficult_;
            box_label.class_label_ = ori_labels[i].class_label_;
            box_label.box_[0] = float(ori_x - rand_x) / float(rand_w);
            box_label.box_[1] = float(ori_y - rand_y) / float(rand_h);
            box_label.box_[2] = float(ori_w) / float(rand_w);
            box_label.box_[3] = float(ori_h) / float(rand_h);
            // int xmin = std::max(ori_x - ori_w / 2, rand_x);
            // int ymin = std::max(ori_y - ori_h / 2, rand_y);
            // int xmax = std::min(ori_x + ori_w / 2, rand_x + rand_w);
            // int ymax = std::min(ori_y + ori_h / 2, rand_y + rand_h);
            // if (xmin > xmax || ymin > ymax) {
            //   continue;
            // }
            // box_label.box_[0] = float(xmin + (xmax - xmin) / 2) / float(rand_w);
            // box_label.box_[1] = float(ymin + (ymax - ymin) / 2) / float(rand_h);
            // box_label.box_[2] = float(xmax - xmin) / float(rand_w);
            // box_label.box_[3] = float(ymax - ymin) / float(rand_h);
            if (mirror) {
              box_label.box_[0] = std::max(0., 1. - box_label.box_[0]);
              box_label.box_[1] = std::max(0., 1. - box_label.box_[1]);
            }
            box_labels->push_back(box_label);
          }
          if (box_labels->size() > 0) {
            cv::Rect roi(rand_x, rand_y, rand_w, rand_h);
            cv_rand_img = cv_img(roi);
            if (mirror) {
              cv::flip(cv_rand_img, cv_rand_img, 1); // horizen flip
            }
          }
        }
        cv::resize(cv_rand_img, cv_rand_img, cv::Size(img_width, img_height));
        // Transform the cv::image into blob.
        Transform(cv_rand_img, transformed_blob);
        return;
  }
  // tests only, TODO: clean
  void Transform(const Datum& datum, TBlob<Dtype>* transformed_blob) const {
    cv::Mat img;
    DatumToCVMat(datum, img, false);
    Transform(img, transformed_blob);
  }

  void Fill3Randoms(unsigned int *rand) const;

  void TransformInv(const Blob* blob, vector<cv::Mat>* cv_imgs);
  void TransformInv(const Dtype* data, cv::Mat* cv_img, const int height,
      const int width, const int channels);

  vector<int> InferBlobShape(const cv::Mat& cv_img);
  vector<int> InferDatumShape(const Datum& datum);
  vector<int> InferCVMatShape(const cv::Mat& img);

  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param bottom_shape
   *    The shape of the data to be transformed.
   */
  vector<int> InferBlobShape(const vector<int>& bottom_shape, bool use_gpu = false);

  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const Datum& datum);

  /**
   * @brief Crops the datum according to bbox.
   */

  void CropImage(const Datum& datum, const NormalizedBBox& bbox, Datum* crop_datum);

  /**
   * @brief Crops the datum and AnnotationGroup according to bbox.
   */
  void CropImage(const AnnotatedDatum& anno_datum, const NormalizedBBox& bbox,
                 AnnotatedDatum* cropped_anno_datum);

  /**
   * @brief Expand the datum.
   */
  void ExpandImage(const Datum& datum, const float expand_ratio,
                   NormalizedBBox* expand_bbox, Datum* expanded_datum);

  /**
   * @brief Expand the datum and adjust AnnotationGroup.
   */
  void ExpandImage(const AnnotatedDatum& anno_datum, AnnotatedDatum* expanded_anno_datum);

  /**
   * @brief Apply distortion to the datum.
   */
  void DistortImage(const Datum& datum, Datum* distort_datum);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the annotated data.
   *
   * @param anno_datum
   *    AnnotatedDatum containing the data and annotation to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See annotated_data_layer.cpp for an example.
   * @param transformed_anno_vec
   *    This is destination annotation.
   */
  void Transform(const AnnotatedDatum& anno_datum,
                 TBlob<Dtype>* transformed_blob,
                 RepeatedPtrField<AnnotationGroup>* transformed_anno_vec);

  void Transform(const AnnotatedDatum& anno_datum,
                 TBlob<Dtype>* transformed_blob,
                 RepeatedPtrField<AnnotationGroup>* transformed_anno_vec,
                 bool* do_mirror);

  void Transform(const AnnotatedDatum& anno_datum,
                 TBlob<Dtype>* transformed_blob,
                 vector<AnnotationGroup>* transformed_anno_vec,
                 bool* do_mirror);

  void Transform(const AnnotatedDatum& anno_datum,
                 TBlob<Dtype>* transformed_blob,
                 vector<AnnotationGroup>* transformed_anno_vec);

  bool image_random_resize_enabled() const;
  bool image_center_crop_enabled() const;
  bool image_random_crop_enabled() const;
  void image_random_resize(const cv::Mat& src, cv::Mat& dst) const;
  void image_center_crop(int crop_w, int crop_h, cv::Mat& img) const;
  void image_random_crop(int crop_w, int crop_h, cv::Mat& img) const;

 protected:
  void apply_mean_scale_mirror(const cv::Mat& src, cv::Mat& dst) const;

  void TransformV1(const Datum& datum, Dtype* buf, size_t buf_len);

  unsigned int Rand() const;
  float Rand(float lo, float up) const;

  void Copy(const Datum& datum, Dtype* data, size_t& out_sizeof_element);
  void Copy(const cv::Mat& datum, Dtype* data);

  /**
   * @brief Transform the annotation according to the transformation applied
   * to the datum.
   *
   * @param anno_datum
   *    AnnotatedDatum containing the data and annotation to be transformed.
   * @param do_resize
   *    If true, resize the annotation accordingly before crop.
   * @param crop_bbox
   *    The cropped region applied to anno_datum.datum()
   * @param do_mirror
   *    If true, meaning the datum has mirrored.
   * @param transformed_anno_group_all
   *    Stores all transformed AnnotationGroup.
   */
  void TransformAnnotation(
      const AnnotatedDatum& anno_datum, const bool do_resize,
      const NormalizedBBox& crop_bbox, const bool do_mirror,
      RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all);


  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_data_layer.cpp for an example.
   */
  void Transform(const cv::Mat& cv_img, TBlob<Dtype>* transformed_blob,
                 NormalizedBBox* crop_bbox, bool* do_mirror);

  /**
   * @brief Crops img according to bbox.
   */
  void CropImage(const cv::Mat& img, const NormalizedBBox& bbox, cv::Mat* crop_img);

  /**
   * @brief Expand img to include mean value as background.
   */
  void ExpandImage(const cv::Mat& img, const float expand_ratio,
                   NormalizedBBox* expand_bbox, cv::Mat* expand_img);

  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const vector<Datum> & datum_vector);

  void Transform(const Datum& datum,
      Dtype *transformed_data, const std::array<unsigned int, 3>& rand);

 protected:
  // Transform and return the transformation information.
  void Transform(const Datum& datum, Dtype* transformed_data,
                 NormalizedBBox* crop_bbox, bool* do_mirror);
  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data and return transform information.
   */
  void Transform(const Datum& datum, TBlob<Dtype>* transformed_blob,
                 NormalizedBBox* crop_bbox, bool* do_mirror);

  // Tranformation parameters
  TransformationParameter param_;
  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
  TBlob<float> data_mean_;
  vector<float> mean_values_;
  cv::Mat mean_mat_orig_;
  mutable cv::Mat mean_mat_;
  mutable cv::Mat tmp_;

  const float rand_resize_ratio_lower_, rand_resize_ratio_upper_;
  const float vertical_stretch_lower_;
  const float vertical_stretch_upper_;
  const float horizontal_stretch_lower_;
  const float horizontal_stretch_upper_;
  const bool allow_upscale_;
  GPUMemory::Workspace mean_values_gpu_;

  static constexpr double UM = static_cast<double>(UINT_MAX);
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_
