#pragma once
#include "layers.h"

namespace dnn
{
    // the main vgg building block
    template<int num_filters, template<typename> class BN, typename SUBNET>
    using vgg_block = relu<BN<con<num_filters, 3, 3, 1, 1, SUBNET>>>;

    // the main blocks for VGG without batch norm
    template<typename SUBNET>
    using vgg_block512 = vgg_block<512, affine, SUBNET>;
    template<typename SUBNET>
    using vgg_block256 = vgg_block<256, affine, SUBNET>;
    template<typename SUBNET>
    using vgg_block128 = vgg_block<128, affine, SUBNET>;
    template<typename SUBNET>
    using vgg_block64 = vgg_block<64, affine, SUBNET>;

    // the main blocks for VGG with batch norm
    template<typename SUBNET>
    using vgg_bn_block512 = vgg_block<512, bn_con, SUBNET>;
    template<typename SUBNET>
    using vgg_bn_block256 = vgg_block<256, bn_con, SUBNET>;
    template<typename SUBNET>
    using vgg_bn_block128 = vgg_block<128, bn_con, SUBNET>;
    template<typename SUBNET>
    using vgg_bn_block64 = vgg_block<64, bn_con, SUBNET>;

    // the final fully connected layers of VGG
    template<typename SUBNET> using vgg_fc = fc<1000, fc<4096, fc<4096, SUBNET>>>;

    // VGG 11
    template<typename SUBNET>
    using vgg11_backbone =
        max_pool<2, 2, 2, 2, repeat<2, vgg_block512,
        max_pool<2, 2, 2, 2, repeat<2, vgg_block512,
        max_pool<2, 2, 2, 2, repeat<2, vgg_block256,
        max_pool<2, 2, 2, 2, repeat<1, vgg_block128,
        max_pool<2, 2, 2, 2, repeat<1, vgg_block64,
        SUBNET>>>>>>>>>>;
    using vgg11_t = loss_multiclass_log<
        vgg_fc<vgg11_backbone<input<input_rgb_image>>>>;

    // VGG 11 with batch normalization
    template<typename SUBNET>
    using vgg11_bn_backbone =
        max_pool<2, 2, 2, 2, repeat<2, vgg_bn_block512,
        max_pool<2, 2, 2, 2, repeat<2, vgg_bn_block512,
        max_pool<2, 2, 2, 2, repeat<2, vgg_bn_block256,
        max_pool<2, 2, 2, 2, repeat<2, vgg_bn_block128,
        max_pool<2, 2, 2, 2, repeat<2, vgg_bn_block64,
        SUBNET>>>>>>>>>>;
    using vgg11_bn_t = loss_multiclass_log<
        vgg_fc<vgg11_bn_backbone<input<input_rgb_image>>>>;

    // VGG 13
    template<typename SUBNET>
    using vgg13_backbone =
        max_pool<2, 2, 2, 2, repeat<2, vgg_block512,
        max_pool<2, 2, 2, 2, repeat<2, vgg_block512,
        max_pool<2, 2, 2, 2, repeat<2, vgg_block256,
        max_pool<2, 2, 2, 2, repeat<2, vgg_block128,
        max_pool<2, 2, 2, 2, repeat<2, vgg_block64,
        SUBNET>>>>>>>>>>;
    using vgg13_t = loss_multiclass_log<
        vgg_fc<vgg13_backbone<input<input_rgb_image>>>>;

    // VGG 13 with batch normalization
    template<typename SUBNET>
    using vgg13_bn_backbone =
        max_pool<2, 2, 2, 2, repeat<2, vgg_bn_block512,
        max_pool<2, 2, 2, 2, repeat<2, vgg_bn_block512,
        max_pool<2, 2, 2, 2, repeat<2, vgg_bn_block256,
        max_pool<2, 2, 2, 2, repeat<2, vgg_bn_block128,
        max_pool<2, 2, 2, 2, repeat<2, vgg_bn_block64,
        SUBNET>>>>>>>>>>;
    using vgg13_bn_t = loss_multiclass_log<
        vgg_fc<vgg13_bn_backbone<input<input_rgb_image>>>>;

    // VGG 16
    template<typename SUBNET>
    using vgg16_backbone =
        max_pool<2, 2, 2, 2, repeat<3, vgg_block512,
        max_pool<2, 2, 2, 2, repeat<3, vgg_block512,
        max_pool<2, 2, 2, 2, repeat<3, vgg_block256,
        max_pool<2, 2, 2, 2, repeat<2, vgg_block128,
        max_pool<2, 2, 2, 2, repeat<2, vgg_block64,
        SUBNET>>>>>>>>>>;
    using vgg16_t = loss_multiclass_log<
        vgg_fc<vgg16_backbone<input<input_rgb_image>>>>;

    // VGG 16 with batch normalization
    template<typename SUBNET>
    using vgg16_bn_backbone =
        max_pool<2, 2, 2, 2, repeat<3, vgg_bn_block512,
        max_pool<2, 2, 2, 2, repeat<3, vgg_bn_block512,
        max_pool<2, 2, 2, 2, repeat<3, vgg_bn_block256,
        max_pool<2, 2, 2, 2, repeat<2, vgg_bn_block128,
        max_pool<2, 2, 2, 2, repeat<2, vgg_bn_block64,
        SUBNET>>>>>>>>>>;
    using vgg16_bn_t = loss_multiclass_log<
        vgg_fc<vgg16_bn_backbone<input<input_rgb_image>>>>;

    // VGG 19
    template<typename SUBNET>
    using vgg19_backbone =
        max_pool<2, 2, 2, 2, repeat<4, vgg_block512,
        max_pool<2, 2, 2, 2, repeat<4, vgg_block512,
        max_pool<2, 2, 2, 2, repeat<4, vgg_block256,
        max_pool<2, 2, 2, 2, repeat<2, vgg_block128,
        max_pool<2, 2, 2, 2, repeat<2, vgg_block64,
        SUBNET>>>>>>>>>>;
    using vgg19_t = loss_multiclass_log<
        vgg_fc<vgg19_backbone<input<input_rgb_image>>>>;

    // VGG 19 with batch normalization
    template<typename SUBNET>
    using vgg19_bn_backbone =
        max_pool<2, 2, 2, 2, repeat<4, vgg_bn_block512,
        max_pool<2, 2, 2, 2, repeat<4, vgg_bn_block512,
        max_pool<2, 2, 2, 2, repeat<4, vgg_bn_block256,
        max_pool<2, 2, 2, 2, repeat<2, vgg_bn_block128,
        max_pool<2, 2, 2, 2, repeat<2, vgg_bn_block64,
        SUBNET>>>>>>>>>>;
    using vgg19_bn_t = loss_multiclass_log<
        vgg_fc<vgg19_bn_backbone<input<input_rgb_image>>>>;
}
