#include <iostream>

#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>

#include "resnet.h"
#include "training.h"

const std::vector<std::string> models =
{
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
};

void print_models()
{
    std::cout << "List of available models:" << std::endl;
    for (const auto& model : models)
    {
        std::cout << "  - " << model << std::endl;
    }
}

template<typename NET>
void print_accuracy(
    NET net,
    const std::string& mode,
    const std::vector<dlib::matrix<unsigned char>>& images,
    const std::vector<unsigned long>& labels
    )
{
    auto predicted_labels = net(images);
    int num_right = 0;
    int num_wrong = 0;
    for (size_t i = 0; i < images.size(); i++)
    {
        if (predicted_labels[i] == labels[i])
            ++num_right;
        else
            ++num_wrong;
    }
    std::cout << mode << " num_right: " << num_right << std::endl;
    std::cout << mode << " num_wrong: " << num_wrong << std::endl;
    std::cout << mode << " accuracy: " << num_right / static_cast<double>(num_right + num_wrong) << std::endl;

}

template<typename NET>
void train_network(
    NET net,
    unsigned long mini_batch_size,
    const std::string& model_name,
    const std::vector<dlib::matrix<unsigned char>>& training_images,
    const std::vector<unsigned long>& training_labels,
    const std::vector<dlib::matrix<unsigned char>>& testing_images,
    const std::vector<unsigned long>& testing_labels
)
{
    dnn::dnn_trainer<decltype(net)> trainer(net);
    trainer.set_synchronization_file(model_name + "_sync", std::chrono::minutes(5));
    trainer.set_mini_batch_size(mini_batch_size);
    trainer.set_max_num_epochs(10);
    std::cout << trainer << std::endl;
    trainer.be_verbose();
    trainer.train(training_images, training_labels);
    trainer.get_net();
    dnn::serialize(model_name + "_mnist.dnn") << net;
    print_accuracy(net, "training", training_images, training_labels);
    print_accuracy(net, "testing", testing_images, testing_labels);
}

int main(int argc, char** argv) try
{
    dlib::command_line_parser parser;
    parser.add_option("model", "the network architecture to use", 1);
    parser.add_option("mnist-root", "path to the mnist root (default: mnist)", 1);
    parser.add_option("mini-batch-size", "mini batch size (default: 32)", 1);
    parser.set_group_name("Help");
    parser.add_option("h", "");
    parser.add_option("help", "display this message and exit");
    parser.add_option("list-models", "print the implemented network architectures");
    parser.parse(argc, argv);

    if (parser.option("help") || parser.option("h"))
    {
        parser.print_options();
        return EXIT_SUCCESS;
    }

    if (parser.option("list-models"))
    {
        print_models();
        return EXIT_SUCCESS;
    }

    if (!parser.option("model"))
    {
        std::cout << "\nPlease, specify at least one model to use." << std::endl;
        std::cout << "Use --list-models or --help for more information.\n" << std::endl;
        return EXIT_FAILURE;
    }

    std::string model_name = parser.option("model").argument();
    auto model_idx = std::find(models.begin(), models.end(), model_name);
    if (model_idx == models.end())
    {
        std::cout << "Model \"" << model_name << "\" is unsupported." << std::endl;
        std::cout << "Use --list-models for a list of all supported network architectures." << std::endl;
        return EXIT_FAILURE;
    }

    // load the MNIST dataset
    std::string mnist_root = dlib::get_option(parser, "mnist-root", "mnist");
    std::vector<dlib::matrix<unsigned char>> training_images;
    std::vector<unsigned long>         training_labels;
    std::vector<dlib::matrix<unsigned char>> testing_images;
    std::vector<unsigned long>         testing_labels;
    dlib::load_mnist_dataset(mnist_root, training_images, training_labels, testing_images, testing_labels);

    unsigned long mini_batch_size = dlib::get_option(parser, "mini-batch-size", 32);
    if (model_name == "resnet18")
    {
        using net_type = dnn::loss_multiclass_log<
            dnn::fc<10,
            dnn::resnet18_backbone<
            dnn::input<dlib::matrix<unsigned char>>
            >>>;
        net_type resnet;
        train_network(resnet, mini_batch_size, model_name, training_images, training_labels, testing_images, testing_labels);
    }
    else if (model_name == "resnet34")
    {
        using net_type = dnn::loss_multiclass_log<
            dnn::fc<10,
            dnn::resnet34_backbone<
            dnn::input<dlib::matrix<unsigned char>>
            >>>;
        net_type resnet;
        train_network(resnet, mini_batch_size, model_name, training_images, training_labels, testing_images, testing_labels);
    }
    else if (model_name == "resnet50")
    {
        using net_type = dnn::loss_multiclass_log<
            dnn::fc<10,
            dnn::resnet50_backbone<
            dnn::input<dlib::matrix<unsigned char>>
            >>>;
        net_type resnet;
        train_network(resnet, mini_batch_size, model_name, training_images, training_labels, testing_images, testing_labels);
    }
    else if (model_name == "resnet101")
    {
        using net_type = dnn::loss_multiclass_log<
            dnn::fc<10,
            dnn::resnet101_backbone<
            dnn::input<dlib::matrix<unsigned char>>
            >>>;
        net_type resnet;
        train_network(resnet, mini_batch_size, model_name, training_images, training_labels, testing_images, testing_labels);
    }
    else if (model_name == "resnet152")
    {
        using net_type = dnn::loss_multiclass_log<
            dnn::fc<10,
            dnn::resnet152_backbone<
            dnn::input<dlib::matrix<unsigned char>>
            >>>;
        net_type resnet;
        train_network(resnet, mini_batch_size, model_name, training_images, training_labels, testing_images, testing_labels);
    }

    return EXIT_SUCCESS;
}
catch (std::exception& e)
{
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
}
