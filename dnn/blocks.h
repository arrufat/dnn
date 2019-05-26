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
    using basicblock = BN<conv<num_filters, 3, 1, 1, relu<BN<conv<num_filters, 3, stride, 1, SUBNET>>>>>;

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
    using residual_down = add_prev2<avg_pool<2, 2, 1, skip1<tag2<BLOCK<num_filters, BN, 2, tag1<SUBNET>>>>>>;

    // residual block
    template<
        template<int, template<typename> class, int, typename> class BLOCK,
        int num_filters,
        template<typename> class BN,
        typename SUBNET
    >
    using resblock = relu<residual<BLOCK, num_filters, BN, SUBNET>>;
    // residual block
    template<
        template<int, template<typename> class, int, typename> class BLOCK,
        int num_filters,
        template<typename> class BN,
        typename SUBNET
    >
    using resblock_down = relu<residual_down<BLOCK, num_filters, BN, SUBNET>>;

    template<int num_filters, typename SUBNET>
    using resbasicblock_down = resblock_down<basicblock, num_filters, bn_con, SUBNET>;
    template<int num_filters, typename SUBNET>
    using resbottleneck_down = resblock_down<bottleneck, num_filters, affine, SUBNET>;
    template<int num_filters, typename SUBNET>
    using aresbasicblock_down = resblock_down<basicblock, num_filters, affine, SUBNET>;
    template<int num_filters, typename SUBNET>
    using aresbottleneck_down = resblock_down<bottleneck, num_filters, bn_con, SUBNET>;

    // some useful definitions to allow the use of the repeat layer
    template<typename SUBNET> using resbasicblock512 = resblock<basicblock, 512, bn_con, SUBNET>;
    template<typename SUBNET> using resbasicblock256 = resblock<basicblock, 256, bn_con, SUBNET>;
    template<typename SUBNET> using resbasicblock128 = resblock<basicblock, 128, bn_con, SUBNET>;
    template<typename SUBNET> using resbasicblock64 = resblock<basicblock, 64, bn_con, SUBNET>;
    template<typename SUBNET> using resbottleneck512 = resblock<bottleneck, 512, bn_con, SUBNET>;
    template<typename SUBNET> using resbottleneck256 = resblock<bottleneck, 256, bn_con, SUBNET>;
    template<typename SUBNET> using resbottleneck128 = resblock<bottleneck, 128, bn_con, SUBNET>;
    template<typename SUBNET> using resbottleneck64 = resblock<bottleneck, 64, bn_con, SUBNET>;

    // and the equivalent affine versions for inference
    template<typename SUBNET> using aresbasicblock512 = resblock<basicblock, 512, affine, SUBNET>;
    template<typename SUBNET> using aresbasicblock256 = resblock<basicblock, 256, affine, SUBNET>;
    template<typename SUBNET> using aresbasicblock128 = resblock<basicblock, 128, affine, SUBNET>;
    template<typename SUBNET> using aresbasicblock64 = resblock<basicblock, 64, affine, SUBNET>;
    template<typename SUBNET> using aresbottleneck512 = resblock<bottleneck, 512, affine, SUBNET>;
    template<typename SUBNET> using aresbottleneck256 = resblock<bottleneck, 256, affine, SUBNET>;
    template<typename SUBNET> using aresbottleneck128 = resblock<bottleneck, 128, affine, SUBNET>;
    template<typename SUBNET> using aresbottleneck64 = resblock<bottleneck, 64, affine, SUBNET>;
}
