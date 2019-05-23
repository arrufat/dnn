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
    using block = BN<conv<num_filters, 3, 1, 1, relu<BN<conv<num_filters, 3, stride, 1, SUBNET>>>>>;

    // the resnet bottleneck block
    template<
        int num_filters,
        template<typename> class BN,  // some kind of batch normalization or affine layer
        int stride,
        typename SUBNET
    >
    using bottleneck = bn_con<conv<4 * num_filters, 1, 1, 0, BN<conv<num_filters, 3, 1, 1, BN<conv<num_filters, 1, 1, 0, SUBNET>>>>>>;

    // a residual making use of the skip layer mechanism
    template<
        template<int, template<typename> class, int, typename> class block,  // a basic or bottleneck block defined before
        int num_filters,
        template<typename> class BN,  // some kind of batch normalization or affine layer
        typename SUBNET
    > // adds the block to the result of tag1 (the subnet)
    using residual = add_prev1<block<num_filters, BN, 1, tag1<SUBNET>>>;

    // a residual that does subsampling (we need to subsample the output of the subnet, too
    template<
        template<int, template<typename> class, int, typename> class block,  // a basic or bottleneck block defined before
        int num_filters,
        template<typename> class BN,
        typename SUBNET
    >
    using residual_down = add_prev2<avg_pool<2, 2, 1, skip1<tag2<block<num_filters, BN, 2, tag1<SUBNET>>>>>>;

    // residual block
    template<int num_filters,  typename SUBNET>
    using resblock = relu<residual<block, num_filters, bn_con, SUBNET>>;
    // downsampling residual block
    template<int num_filters, typename SUBNET>
    using resblock_down = relu<residual_down<block, num_filters, bn_con, SUBNET>>;
    // residual block with batch norm replaced with affine for inference
    template<int num_filters, typename SUBNET>
    using aresblock = relu<residual<block, num_filters, affine, SUBNET>>;
    // downsampling residual block with batch norm replaced with affine for inference
    template<int num_filters, typename SUBNET>
    using aresblock_down = relu<residual_down<block, num_filters, affine, SUBNET>>;

    // residual bottleneck
    template<int num_filters,  typename SUBNET>
    using resbottleneck = relu<residual<bottleneck, num_filters, bn_con, SUBNET>>;
    // downsampling residual bottleneck
    template<int num_filters, typename SUBNET>
    using resbottleneck_down = relu<residual_down<bottleneck, num_filters, bn_con, SUBNET>>;
    // residual bottleneck with batch norm replaced with affine for inference
    template<int num_filters, typename SUBNET>
    using aresbottleneck = relu<residual<bottleneck, num_filters, affine, SUBNET>>;
    // downsampling residual bottleneck with batch norm replaced with affine for inference
    template<int num_filters, typename SUBNET>
    using aresbottleneck_down = relu<residual_down<bottleneck, num_filters, affine, SUBNET>>;

    // some useful definitions to allow the use of the repeat layer
    template<typename SUBNET> using resblock512 = resblock<512, SUBNET>;
    template<typename SUBNET> using resblock256 = resblock<256, SUBNET>;
    template<typename SUBNET> using resblock128 = resblock<128, SUBNET>;
    template<typename SUBNET> using resblock64 = resblock<64, SUBNET>;
    template<typename SUBNET> using resbottleneck512 = resbottleneck<512, SUBNET>;
    template<typename SUBNET> using resbottleneck256 = resbottleneck<256, SUBNET>;
    template<typename SUBNET> using resbottleneck128 = resbottleneck<128, SUBNET>;
    template<typename SUBNET> using resbottleneck64 = resbottleneck<64, SUBNET>;

    // and the equivalent affine versions for inference
    template<typename SUBNET> using aresblock512 = aresblock<512, SUBNET>;
    template<typename SUBNET> using aresblock256 = aresblock<256, SUBNET>;
    template<typename SUBNET> using aresblock128 = aresblock<128, SUBNET>;
    template<typename SUBNET> using aresblock64 = aresblock<64, SUBNET>;
    template<typename SUBNET> using aresbottleneck512 = aresbottleneck<512, SUBNET>;
    template<typename SUBNET> using aresbottleneck256 = aresbottleneck<256, SUBNET>;
    template<typename SUBNET> using aresbottleneck128 = aresbottleneck<128, SUBNET>;
    template<typename SUBNET> using aresbottleneck64 = aresbottleneck<64, SUBNET>;
}
