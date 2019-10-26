#pragma once

#include "layers.h"

namespace dnn::resnet
{
    // the resnet basic block
    template<
        long num_filters,
        template<typename> class BN,  // some kind of batch normalization or affine layer
        int stride,
        typename SUBNET
    >
    using basicblock = BN<con<num_filters, 3, 3, 1, 1, relu<BN<con<num_filters, 3, 3, stride, stride, SUBNET>>>>>;

    // the resnet bottleneck block
    template<
        long num_filters,
        template<typename> class BN,  // some kind of batch normalization or affine layer
        int stride,
        typename SUBNET
    >
    using bottleneck = BN<con<4 * num_filters, 1, 1, 1, 1, relu<BN<con<num_filters, 3, 3, stride, stride, relu<BN<con<num_filters, 1, 1, 1, 1, SUBNET>>>>>>>>;

    // a residual making use of the skip layer mechanism
    template<
        template<long, template<typename> class, int, typename> class BLOCK,  // a basic or bottleneck block defined before
        long num_filters,
        template<typename> class BN,  // some kind of batch normalization or affine layer
        typename SUBNET
    > // adds the block to the result of tag1 (the subnet)
    using residual = add_prev1<BLOCK<num_filters, BN, 1, tag1<SUBNET>>>;

    // a residual that does subsampling (we need to subsample the output of the subnet, too)
    template<
        template<long, template<typename> class, int, typename> class BLOCK,  // a basic or bottleneck block defined before
        long num_filters,
        template<typename> class BN,
        typename SUBNET
    >
    using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<BLOCK<num_filters, BN, 2, tag1<SUBNET>>>>>>;

    // residual block with optional downsampling and batch normalization
    template<
        template<template<long, template<typename> class, int, typename> class, long, template<typename>class, typename> class RESIDUAL,
        template<long, template<typename> class, int, typename> class BLOCK,
        long num_filters,
        template<typename> class BN,
        typename SUBNET
    >
    using residual_block = relu<RESIDUAL<BLOCK, num_filters, BN, SUBNET>>;

    namespace train
    {
        template<long num_filters, typename SUBNET>
        using resbasicblock_down = residual_block<residual_down, basicblock, num_filters, bn_con, SUBNET>;
        template<long num_filters, typename SUBNET>
        using resbottleneck_down = residual_block<residual_down, bottleneck, num_filters, bn_con, SUBNET>;

        // some definitions to allow the use of the repeat layer
        template<typename SUBNET> using resbasicblock_512 = residual_block<residual, basicblock, 512, bn_con, SUBNET>;
        template<typename SUBNET> using resbasicblock_256 = residual_block<residual, basicblock, 256, bn_con, SUBNET>;
        template<typename SUBNET> using resbasicblock_128 = residual_block<residual, basicblock, 128, bn_con, SUBNET>;
        template<typename SUBNET> using resbasicblock_64  = residual_block<residual, basicblock,  64, bn_con, SUBNET>;
        template<typename SUBNET> using resbottleneck_512 = residual_block<residual, bottleneck, 512, bn_con, SUBNET>;
        template<typename SUBNET> using resbottleneck_256 = residual_block<residual, bottleneck, 256, bn_con, SUBNET>;
        template<typename SUBNET> using resbottleneck_128 = residual_block<residual, bottleneck, 128, bn_con, SUBNET>;
        template<typename SUBNET> using resbottleneck_64  = residual_block<residual, bottleneck,  64, bn_con, SUBNET>;

        // common processing for standard resnet inputs
        template<typename INPUT>
        using input_processing = max_pool<3, 3, 2, 2, relu<bn_con<con<64, 7, 7, 2, 2, INPUT>>>>;

        // the resnet backbone with basicblocks
        template<long nf_512, long nf_256, long nf_128, long nf_64, typename INPUT>
        using backbone_basicblock =
            repeat<nf_512, resbasicblock_512, resbasicblock_down<512,
            repeat<nf_256, resbasicblock_256, resbasicblock_down<256,
            repeat<nf_128, resbasicblock_128, resbasicblock_down<128,
            repeat<nf_64,  resbasicblock_64,
            input_processing<INPUT>>>>>>>>;

        // the resnet backbone with bottlenecks
        template<long nf_512, long nf_256, long nf_128, long nf_64, typename INPUT>
        using backbone_bottleneck =
            repeat<nf_512, resbottleneck_512, resbottleneck_down<512,
            repeat<nf_256, resbottleneck_256, resbottleneck_down<256,
            repeat<nf_128, resbottleneck_128, resbottleneck_down<128,
            repeat<nf_64,  resbottleneck_64,
            input_processing<INPUT>>>>>>>>;

        // the backbones for the classic architectures
        template<typename INPUT> using backbone_18  = backbone_basicblock<1, 1, 1, 2, INPUT>;
        template<typename INPUT> using backbone_34  = backbone_basicblock<2, 5, 3, 3, INPUT>;
        template<typename INPUT> using backbone_50  = backbone_bottleneck<2, 5, 3, 3, INPUT>;
        template<typename INPUT> using backbone_101 = backbone_bottleneck<2, 22, 3, 3, INPUT>;
        template<typename INPUT> using backbone_152 = backbone_bottleneck<2, 35, 3, 3, INPUT>;

        // the typical classifier models
        using  _18  = loss_multiclass_log<fc<1000, avg_pool_everything<backbone_18<input_rgb_image>>>>;
        using  _34  = loss_multiclass_log<fc<1000, avg_pool_everything<backbone_34<input_rgb_image>>>>;
        using  _50  = loss_multiclass_log<fc<1000, avg_pool_everything<backbone_50<input_rgb_image>>>>;
        using  _101 = loss_multiclass_log<fc<1000, avg_pool_everything<backbone_101<input_rgb_image>>>>;
        using  _152 = loss_multiclass_log<fc<1000, avg_pool_everything<backbone_152<input_rgb_image>>>>;
    }

    namespace infer
    {
        template<long num_filters, typename SUBNET>
        using resbasicblock_down = residual_block<residual_down, basicblock, num_filters, affine, SUBNET>;
        template<long num_filters, typename SUBNET>
        using resbottleneck_down = residual_block<residual_down, bottleneck, num_filters, affine, SUBNET>;

        // some definitions to allow the use of the repeat layer
        template<typename SUBNET> using resbasicblock_512 = residual_block<residual, basicblock, 512, affine, SUBNET>;
        template<typename SUBNET> using resbasicblock_256 = residual_block<residual, basicblock, 256, affine, SUBNET>;
        template<typename SUBNET> using resbasicblock_128 = residual_block<residual, basicblock, 128, affine, SUBNET>;
        template<typename SUBNET> using resbasicblock_64  = residual_block<residual, basicblock,  64, affine, SUBNET>;
        template<typename SUBNET> using resbottleneck_512 = residual_block<residual, bottleneck, 512, affine, SUBNET>;
        template<typename SUBNET> using resbottleneck_256 = residual_block<residual, bottleneck, 256, affine, SUBNET>;
        template<typename SUBNET> using resbottleneck_128 = residual_block<residual, bottleneck, 128, affine, SUBNET>;
        template<typename SUBNET> using resbottleneck_64  = residual_block<residual, bottleneck,  64, affine, SUBNET>;

        // common processing for standard resnet inputs
        template<typename INPUT>
        using input_processing = max_pool<3, 3, 2, 2, relu<affine<con<64, 7, 7, 2, 2, INPUT>>>>;

        // the resnet backbone with basicblocks
        template<long nf_512, long nf_256, long nf_128, long nf_64, typename INPUT>
        using backbone_basicblock =
            repeat<nf_512, resbasicblock_512, resbasicblock_down<512,
            repeat<nf_256, resbasicblock_256, resbasicblock_down<256,
            repeat<nf_128, resbasicblock_128, resbasicblock_down<128,
            repeat<nf_64,  resbasicblock_64,  resbasicblock_down<64,
            input_processing<INPUT>>>>>>>>>;

        // the resnet backbone with bottlenecks
        template<long nf_512, long nf_256, long nf_128, long nf_64, typename INPUT>
        using backbone_bottleneck =
            repeat<nf_512, resbasicblock_512, resbasicblock_down<512,
            repeat<nf_256, resbasicblock_256, resbasicblock_down<256,
            repeat<nf_128, resbasicblock_128, resbasicblock_down<128,
            repeat<nf_64,  resbasicblock_64,  resbasicblock_down<64,
            input_processing<INPUT>>>>>>>>>;

        // the backbones for the classic architectures
        template<typename INPUT> using backbone_18  = backbone_basicblock<1, 1, 1, 2, INPUT>;
        template<typename INPUT> using backbone_34  = backbone_basicblock<2, 5, 3, 3, INPUT>;
        template<typename INPUT> using backbone_50  = backbone_bottleneck<2, 5, 3, 3, INPUT>;
        template<typename INPUT> using backbone_101 = backbone_bottleneck<2, 22, 3, 3, INPUT>;
        template<typename INPUT> using backbone_152 = backbone_bottleneck<2, 35, 3, 3, INPUT>;

        // the typical classifier models
        using  _18  = loss_multiclass_log<fc<1000, avg_pool_everything<backbone_18<input_rgb_image>>>>;
        using  _34  = loss_multiclass_log<fc<1000, avg_pool_everything<backbone_34<input_rgb_image>>>>;
        using  _50  = loss_multiclass_log<fc<1000, avg_pool_everything<backbone_50<input_rgb_image>>>>;
        using  _101 = loss_multiclass_log<fc<1000, avg_pool_everything<backbone_101<input_rgb_image>>>>;
        using  _152 = loss_multiclass_log<fc<1000, avg_pool_everything<backbone_152<input_rgb_image>>>>;
    }
}
