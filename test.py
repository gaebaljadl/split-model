from splitter import *
from sample_data import *
import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Prepare a dict which will help us convert class index into a readable class name.
    class_name_dict = load_class_index_to_name_dict()

    # Load the test data.
    # The provided input images do not have uniform size,
    # so we cannot use batch_size > 1.
    dataset = TestDataset()
    dataloader = DataLoader(dataset, batch_size=1)

    # Load pretrained model.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_pretrained_resnet().to(device)

    # Convert given resnet model into list of layers,
    # then split them in half to construct head/tail models.
    # module_list = splitter.convert_to_module_list(model)
    # split_layer_index = len(module_list) // 2
    # head, tail = splitter.split_module_list(module_list, split_layer_index)

    # Load splitted models (created by running splitter.py)
    head = torch.load("./head.pth").to(device)
    tail = torch.load("./tail.pth").to(device)

    # Turn off dropout and batch normalization.
    model.eval()
    head.eval()
    tail.eval()

    # Print split result.
    # print("----- split result -----")
    # print("# layers:", len(module_list))
    # for i in range(len(module_list)):
    #     print("layer", i, module_list[i])
    print("----- head part -----")
    print(head)
    print("----- tail part -----")
    print(tail)

    # Test accuracy.
    print("----- testing the model with one sample for each class -----")
    num_tested = 0
    num_correct = 0
    with torch.no_grad():
        for image, label in dataloader:
            # Forward pass.
            pred = model(image.to(device))

            split_pred = head(image.to(device))
            split_pred = tail(split_pred)

            # Get the class index of prediction with highest probability.
            max_prob_class = torch.argmax(pred)
            split_max_prob_class = torch.argmax(split_pred)

            # Two predictions must be identical if we did the job correctly.
            assert(max_prob_class.item() == split_max_prob_class.item())

            # Check if the output with highest probability is equal to groundtruth.
            # Note: class_name can contain multiple aliases (ex. "dog, doge")
            #       so we must compare label with 'in' operator (ex. "dog" in "dog, doge")
            # Note: the index 0 is for batch dimension, which has size of 1.
            class_name = class_name_dict[max_prob_class.item()]
            is_correct = label[0] in class_name

            # Record accuracy.
            num_tested += 1
            if is_correct:
                num_correct += 1
    
    # Print final accuracy.
    print("correct predictions: {}/{}".format(num_correct, num_tested))
    print("accuracy:", 100.0 * num_correct / num_tested)