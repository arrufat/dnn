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
    using conv = add_layer<dlib::con_<num_filters, kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;

    // a deconvolution with square parameters and padding setting
    template<long num_filters, long kernel_size, int stride, int padding, typename SUBNET>
    using deconv = dlib::add_layer<dlib::cont_<num_filters, kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;

    // a square max pooling with custom stride
    template<long kernel_size, int stride, int padding, typename SUBNET>
    using max_pool = add_layer<dlib::max_pool_<kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;

    // a square avg pooling with custom stride
    template<long kernel_size, int stride, int padding, typename SUBNET>
    using avg_pool = add_layer<dlib::avg_pool_<kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;

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
    using dlib::loss_multiclass_log;
    // input layers
    using dlib::input, dlib::input_rgb_image, dlib::input_rgb_image_sized;
}
