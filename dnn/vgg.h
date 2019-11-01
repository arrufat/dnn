#pragma once

#include "layers.h"

namespace dnn
{
    // regularization must be bn_con or affine
    template<template<typename> class REGULARIZATION>
    struct vgg
    {
        // the main vgg building block, where REG is bn_con or affine
        template<long num_filters, template<typename> class REG, typename SUBNET>
        using block = relu<REG<con<num_filters, 3, 3, 1, 1, SUBNET>>>;

        // the final fc layers of vgg
        template<long num_outputs, typename SUBNET>
        using final_fc = fc<num_outputs, fc<4096, fc<4096, SUBNET>>>;

        // some definitions to allow the use of the repeat layer
        template<typename SUBNET> using block512 = block<512, REGULARIZATION, SUBNET>;
        template<typename SUBNET> using block256 = block<256, REGULARIZATION, SUBNET>;
        template<typename SUBNET> using block128 = block<128, REGULARIZATION, SUBNET>;
        template<typename SUBNET> using block64 = block<64, REGULARIZATION, SUBNET>;

        // the vgg backbone: we need the multiply layer at the end to make sure the repeat layer
        // works properly (all the inputs must conform to the same inteface)
        template<long nb_512, long nb_256, long nb_128, long nb_64, typename INPUT>
        using backbone =
            max_pool<2, 2, 2, 2, repeat<nb_512, block512,
            max_pool<2, 2, 2, 2, repeat<nb_512, block512,
            max_pool<2, 2, 2, 2, repeat<nb_256, block256,
            max_pool<2, 2, 2, 2, repeat<nb_128, block128,
            max_pool<2, 2, 2, 2, repeat<nb_64, block64,
            multiply<INPUT>>>>>>>>>>>;

        template<typename INPUT> using backbone_11 = backbone<2, 2, 1, 1, INPUT>;
        template<typename INPUT> using backbone_13 = backbone<2, 2, 2, 2, INPUT>;
        template<typename INPUT> using backbone_16 = backbone<3, 3, 2, 2, INPUT>;
        template<typename INPUT> using backbone_19 = backbone<4, 4, 2, 2, INPUT>;

        // common vgg classifiers
        using _11 = loss_multiclass_log<final_fc<1000, backbone_11<input_rgb_image>>>;
        using _13 = loss_multiclass_log<final_fc<1000, backbone_13<input_rgb_image>>>;
        using _16 = loss_multiclass_log<final_fc<1000, backbone_16<input_rgb_image>>>;
        using _19 = loss_multiclass_log<final_fc<1000, backbone_19<input_rgb_image>>>;

    };
}
