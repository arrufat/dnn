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

    // a residual that does subsampling (we need to subsample the output of the subnet, too
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
}
