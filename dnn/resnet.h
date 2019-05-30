#pragma once

#include "layers.h"

namespace dnn
{
    // the resnet basic block
    template<
        int num_filters,
        template<typename> class BN,  // some kind of batch normalization or affine layer
        int stride,
        typename SUBNET
    >
    using basicblock = BN<con<num_filters, 3, 3, 1, 1, relu<BN<con<num_filters, 3, 3, stride, stride, SUBNET>>>>>;

    // the resnet bottleneck block
    template<
        int num_filters,
        template<typename> class BN,  // some kind of batch normalization or affine layer
        int stride,
        typename SUBNET
    >
    using bottleneck = bn_con<con<4 * num_filters, 1, 1, 1, 1, BN<con<num_filters, 3, 3, 1, 1, BN<con<num_filters, 1, 1, 1, 1, SUBNET>>>>>>;

    // a residual making use of the skip layer mechanism
    template<
        template<int, template<typename> class, int, typename> class BLOCK,  // a basic or bottleneck block defined before
        int num_filters,
        template<typename> class BN,  // some kind of batch normalization or affine layer
        typename SUBNET
    > // adds the block to the result of tag1 (the subnet)
    using residual = add_prev1<BLOCK<num_filters, BN, 1, tag1<SUBNET>>>;

    // a residual that does subsampling (we need to subsample the output of the subnet, too)
    template<
        template<int, template<typename> class, int, typename> class BLOCK,  // a basic or bottleneck block defined before
        int num_filters,
        template<typename> class BN,
        typename SUBNET
    >
    using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<BLOCK<num_filters, BN, 2, tag1<SUBNET>>>>>>;

    // residual block with optional downsampling and batch normalization
    template<
        template<template<int, template<typename> class, int, typename> class, int, template<typename>class, typename> class RESIDUAL,
        template<int, template<typename> class, int, typename> class BLOCK,
        int num_filters,
        template<typename> class BN,
        typename SUBNET
    >
    using residual_block = relu<RESIDUAL<BLOCK, num_filters, BN, SUBNET>>;

    template<int num_filters, typename SUBNET>
    using resbasicblock_down = residual_block<residual_down, basicblock, num_filters, bn_con, SUBNET>;
    template<int num_filters, typename SUBNET>
    using resbottleneck_down = residual_block<residual_down, bottleneck, num_filters, bn_con, SUBNET>;
    template<int num_filters, typename SUBNET>
    using aresbasicblock_down = residual_block<residual_down, basicblock, num_filters, affine, SUBNET>;
    template<int num_filters, typename SUBNET>
    using aresbottleneck_down = residual_block<residual_down, bottleneck, num_filters, affine, SUBNET>;

    // some useful definitions to allow the use of the repeat layer
    template<typename SUBNET> using resbasicblock512 = residual_block<residual, basicblock, 512, bn_con, SUBNET>;
    template<typename SUBNET> using resbasicblock256 = residual_block<residual, basicblock, 256, bn_con, SUBNET>;
    template<typename SUBNET> using resbasicblock128 = residual_block<residual, basicblock, 128, bn_con, SUBNET>;
    template<typename SUBNET> using resbasicblock64 = residual_block<residual, basicblock, 64, bn_con, SUBNET>;
    template<typename SUBNET> using resbottleneck512 = residual_block<residual, bottleneck, 512, bn_con, SUBNET>;
    template<typename SUBNET> using resbottleneck256 = residual_block<residual, bottleneck, 256, bn_con, SUBNET>;
    template<typename SUBNET> using resbottleneck128 = residual_block<residual, bottleneck, 128, bn_con, SUBNET>;
    template<typename SUBNET> using resbottleneck64 = residual_block<residual, bottleneck, 64, bn_con, SUBNET>;

    // and the equivalent affine versions for inference
    template<typename SUBNET> using aresbasicblock512 = residual_block<residual, basicblock, 512, affine, SUBNET>;
    template<typename SUBNET> using aresbasicblock256 = residual_block<residual, basicblock, 256, affine, SUBNET>;
    template<typename SUBNET> using aresbasicblock128 = residual_block<residual, basicblock, 128, affine, SUBNET>;
    template<typename SUBNET> using aresbasicblock64 = residual_block<residual, basicblock, 64, affine, SUBNET>;
    template<typename SUBNET> using aresbottleneck512 = residual_block<residual, bottleneck, 512, affine, SUBNET>;
    template<typename SUBNET> using aresbottleneck256 = residual_block<residual, bottleneck, 256, affine, SUBNET>;
    template<typename SUBNET> using aresbottleneck128 = residual_block<residual, bottleneck, 128, affine, SUBNET>;
    template<typename SUBNET> using aresbottleneck64 = residual_block<residual, bottleneck, 64, affine, SUBNET>;

    // common input for standard resnets
    template<typename INPUT>
    using resnet_input = max_pool<3, 3, 2, 2, relu<bn_con<con<64, 7, 7, 2, 2, INPUT>>>>;
    template<typename INPUT>
    using aresnet_input = max_pool<3, 3, 2, 2, relu<affine<con<64, 7, 7, 2, 2, INPUT>>>>;

    // resnet 18
    template<typename SUBNET>
    using resnet18_level1 = resbasicblock512<resbasicblock_down<512, SUBNET>>;
    template<typename SUBNET>
    using resnet18_level2 = resbasicblock256<resbasicblock_down<256, SUBNET>>;
    template<typename SUBNET>
    using resnet18_level3 = resbasicblock128<resbasicblock_down<128, SUBNET>>;
    template<typename SUBNET>
    using resnet18_level4 = repeat<2, resbasicblock64, SUBNET>;
    // the resnet 18 backbone
    template<typename INPUT>
    using resnet18_backbone = avg_pool_everything<
        resnet18_level1<
        resnet18_level2<
        resnet18_level3<
        resnet18_level4<
        resnet_input<INPUT>>>>>>;
    using resnet18_t = loss_multiclass_log<fc<1000, resnet18_backbone<input_rgb_image>>>;

    // resnet 18 affine
    template<typename SUBNET>
    using aresnet18_level1 = aresbasicblock512<aresbasicblock_down<512, SUBNET>>;
    template<typename SUBNET>
    using aresnet18_level2 = aresbasicblock256<aresbasicblock_down<256, SUBNET>>;
    template<typename SUBNET>
    using aresnet18_level3 = aresbasicblock128<aresbasicblock_down<128, SUBNET>>;
    template<typename SUBNET>
    using aresnet18_level4 = repeat<2, aresbasicblock64, SUBNET>;
    // the resnet 18 backbone
    template<typename INPUT>
    using aresnet18_backbone = avg_pool_everything<
        aresnet18_level1<
        aresnet18_level2<
        aresnet18_level3<
        aresnet18_level4<
        aresnet_input<INPUT>>>>>>;
    using aresnet18_t = loss_multiclass_log<fc<1000, aresnet18_backbone<input_rgb_image>>>;

    // resnet 34
    template<typename SUBNET>
    using resnet34_level1 = repeat<2, resbasicblock512, resbasicblock_down<512, SUBNET>>;
    template<typename SUBNET>
    using resnet34_level2 = repeat<5, resbasicblock256, resbasicblock_down<256, SUBNET>>;
    template<typename SUBNET>
    using resnet34_level3 = repeat<3, resbasicblock128, resbasicblock_down<128, SUBNET>>;
    template<typename SUBNET>
    using resnet34_level4 = repeat<3, resbasicblock64, SUBNET>;
    // the resnet 34 backbone
    template<typename INPUT>
    using resnet34_backbone = avg_pool_everything<
        resnet34_level1<
        resnet34_level2<
        resnet34_level3<
        resnet34_level4<
        resnet_input<INPUT>>>>>>;
    using resnet34_t = loss_multiclass_log<fc<1000, resnet34_backbone<input_rgb_image>>>;

    // resnet 34 affine
    template<typename SUBNET>
    using aresnet34_level1 = repeat<2, aresbasicblock512, aresbasicblock_down<512, SUBNET>>;
    template<typename SUBNET>
    using aresnet34_level2 = repeat<5, aresbasicblock256, aresbasicblock_down<256, SUBNET>>;
    template<typename SUBNET>
    using aresnet34_level3 = repeat<3, aresbasicblock128, aresbasicblock_down<128, SUBNET>>;
    template<typename SUBNET>
    using aresnet34_level4 = repeat<3, aresbasicblock64, SUBNET>;
    // the resnet 34 backbone
    template<typename INPUT>
    using aresnet34_backbone = avg_pool_everything<
        aresnet34_level1<
        aresnet34_level2<
        aresnet34_level3<
        aresnet34_level4<
        aresnet_input<INPUT>>>>>>;
    using aresnet34_t = loss_multiclass_log<fc<1000, aresnet34_backbone<input_rgb_image>>>;

    // resnet 50
    template<typename SUBNET>
    using resnet50_level1 = repeat<2, resbottleneck512, resbottleneck_down<512, SUBNET>>;
    template<typename SUBNET>
    using resnet50_level2 = repeat<5, resbottleneck256, resbottleneck_down<256, SUBNET>>;
    template<typename SUBNET>
    using resnet50_level3 = repeat<3, resbottleneck128, resbottleneck_down<128, SUBNET>>;
    template<typename SUBNET>
    using resnet50_level4 = repeat<3, resbottleneck64, SUBNET>;
    // the resnet 50 backbone
    template<typename INPUT>
    using resnet50_backbone = avg_pool_everything<
        resnet50_level1<
        resnet50_level2<
        resnet50_level3<
        resnet50_level4<
        resnet_input<INPUT>>>>>>;
    using resnet50_t = loss_multiclass_log<fc<1000, resnet50_backbone<input_rgb_image>>>;

    // resnet 50 affine
    template<typename SUBNET>
    using aresnet50_level1 = repeat<2, aresbottleneck512, aresbottleneck_down<512, SUBNET>>;
    template<typename SUBNET>
    using aresnet50_level2 = repeat<5, aresbottleneck256, aresbottleneck_down<256, SUBNET>>;
    template<typename SUBNET>
    using aresnet50_level3 = repeat<3, aresbottleneck128, aresbottleneck_down<128, SUBNET>>;
    template<typename SUBNET>
    using aresnet50_level4 = repeat<3, aresbottleneck64, SUBNET>;
    // the resnet 50 backbone
    template<typename INPUT>
    using aresnet50_backbone = avg_pool_everything<
        aresnet50_level1<
        aresnet50_level2<
        aresnet50_level3<
        aresnet50_level4<
        aresnet_input<INPUT>>>>>>;
    using aresnet50_t = loss_multiclass_log<fc<1000, aresnet50_backbone<input_rgb_image>>>;

    // resnet 101
    template<typename SUBNET>
    using resnet101_level1 = repeat<2, resbottleneck512, resbottleneck_down<512, SUBNET>>;
    template<typename SUBNET>
    using resnet101_level2 = repeat<22, resbottleneck256, resbottleneck_down<256, SUBNET>>;
    template<typename SUBNET>
    using resnet101_level3 = repeat<3, resbottleneck128, resbottleneck_down<128, SUBNET>>;
    template<typename SUBNET>
    using resnet101_level4 = repeat<3, resbottleneck64, SUBNET>;
    // the resnet 101 backbone
    template<typename INPUT>
    using resnet101_backbone = avg_pool_everything<
        resnet101_level1<
        resnet101_level2<
        resnet101_level3<
        resnet101_level4<
        resnet_input<INPUT>>>>>>;
    using resnet101_t = loss_multiclass_log<fc<1000, resnet101_backbone<input_rgb_image>>>;

    // resnet 101 affine
    template<typename SUBNET>
    using aresnet101_level1 = repeat<2, aresbottleneck512, aresbottleneck_down<512, SUBNET>>;
    template<typename SUBNET>
    using aresnet101_level2 = repeat<22, aresbottleneck256, aresbottleneck_down<256, SUBNET>>;
    template<typename SUBNET>
    using aresnet101_level3 = repeat<3, aresbottleneck128, aresbottleneck_down<128, SUBNET>>;
    template<typename SUBNET>
    using aresnet101_level4 = repeat<3, aresbottleneck64, SUBNET>;
    // the resnet 101 backbone
    template<typename INPUT>
    using aresnet101_backbone = avg_pool_everything<
        aresnet101_level1<
        aresnet101_level2<
        aresnet101_level3<
        aresnet101_level4<
        aresnet_input<INPUT>>>>>>;
    using aresnet101_t = loss_multiclass_log<fc<1000, aresnet101_backbone<input_rgb_image>>>;

    // resnet 152
    template<typename SUBNET>
    using resnet152_level1 = repeat<2, resbottleneck512, resbottleneck_down<512, SUBNET>>;
    template<typename SUBNET>
    using resnet152_level2 = repeat<35, resbottleneck256, resbottleneck_down<256, SUBNET>>;
    template<typename SUBNET>
    using resnet152_level3 = repeat<7, resbottleneck128, resbottleneck_down<128, SUBNET>>;
    template<typename SUBNET>
    using resnet152_level4 = repeat<3, resbottleneck64, SUBNET>;
    // the resnet 152 backbone
    template<typename INPUT>
    using resnet152_backbone = avg_pool_everything<
        resnet152_level1<
        resnet152_level2<
        resnet152_level3<
        resnet152_level4<
        resnet_input<INPUT>>>>>>;
    using resnet152_t = loss_multiclass_log<fc<1000, resnet152_backbone<input_rgb_image>>>;

    // resnet 152 affine
    template<typename SUBNET>
    using aresnet152_level1 = repeat<2, aresbottleneck512, aresbottleneck_down<512, SUBNET>>;
    template<typename SUBNET>
    using aresnet152_level2 = repeat<35, aresbottleneck256, aresbottleneck_down<256, SUBNET>>;
    template<typename SUBNET>
    using aresnet152_level3 = repeat<7, aresbottleneck128, aresbottleneck_down<128, SUBNET>>;
    template<typename SUBNET>
    using aresnet152_level4 = repeat<3, aresbottleneck64, SUBNET>;
    // the resnet 152 backbone
    template<typename INPUT>
    using aresnet152_backbone = avg_pool_everything<
        aresnet152_level1<
        aresnet152_level2<
        aresnet152_level3<
        aresnet152_level4<
        aresnet_input<INPUT>>>>>>;
    using aresnet152_t = loss_multiclass_log<fc<1000, aresnet152_backbone<input_rgb_image>>>;
}
