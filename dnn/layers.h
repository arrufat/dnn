#pragma once

#include <dlib/dnn.h>

namespace dnn
{
    // the add layer mechanism
    using dlib::add_layer;

    // fully connected layer
    using dlib::fc;

    // a convolution with square parameters and padding setting
    template<long num_filters, long kernel_size, int stride, int padding, typename SUBNET>
    using conp = add_layer<dlib::con_<num_filters, kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;
    using dlib::con;

    // a deconvolution with square parameters and padding setting
    template<long num_filters, long kernel_size, int stride, int padding, typename SUBNET>
    using contp = dlib::add_layer<dlib::cont_<num_filters, kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;
    using dlib::cont;

    // a square max pooling with custom stride
    template<long kernel_size, int stride, int padding, typename SUBNET>
    using max_poolp = add_layer<dlib::max_pool_<kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;
    using dlib::max_pool;

    // a square avg pooling with custom stride
    template<long kernel_size, int stride, int padding, typename SUBNET>
    using avg_poolp = add_layer<dlib::avg_pool_<kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;
    using dlib::avg_pool;

    // average and max pooling for everything
    using dlib::avg_pool_everything, dlib::max_pool_everything;
    // batch normalization and affine
    using dlib::bn_con, dlib::bn_fc, dlib::affine;
    // activations
    using dlib::relu;
    // tag operations
    using dlib::tag1, dlib::tag2;
    // skip tagged layers
    using dlib::skip1, dlib::skip2;
    // add previous tagged layers
    using dlib::add_prev1, dlib::add_prev2;
    // layer repetition
    using dlib::repeat;
    // loss functions
    using dlib::loss_binary_hinge,
          dlib::loss_binary_log,
          dlib::loss_dot,
          dlib::loss_epsilon_insensitive,
          dlib::loss_mean_squared,
          dlib::loss_mean_squared_multioutput,
          dlib::loss_mean_squared_per_pixel,
          dlib::loss_metric,
          dlib::loss_mmod,
          dlib::loss_multiclass_log,
          dlib::loss_multiclass_log_per_pixel,
          dlib::loss_multiclass_log_per_pixel_weighted,
          dlib::loss_multimulticlass_log,
          dlib::loss_ranking;
    // input layers
    using dlib::input, dlib::input_rgb_image, dlib::input_rgb_image_sized;
}
