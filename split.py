import sample_data
import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet, resnet50, ResNet50_Weights

class ResNetOutputLayer(torch.nn.Module):
    def __init__(self, avgpool, fc):
        super().__init__()

        self.avgpool = avgpool
        self.fc = fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def split_resnet(resnet_model: ResNet):
    # TODO: split model into two parts
    # plan:
    # 1. manually construct a list of layers from resnet model
    # 2. specify the layer to split (ex. index)
    # 3. run both models and check if the result is same (no accuracy loss)
    layers = torch.nn.ModuleList()
    layers.append(resnet_model.conv1)
    layers.append(resnet_model.bn1)
    layers.append(resnet_model.relu)
    layers.append(resnet_model.maxpool)

    layers.append(resnet_model.layer1)
    layers.append(resnet_model.layer2)
    layers.append(resnet_model.layer3)
    layers.append(resnet_model.layer4)
    
    layers.append(ResNetOutputLayer(resnet_model.avgpool, resnet_model.fc))
    return layers


if __name__ == "__main__":
    # Prepare a dict which will help us convert class index into a readable class name.
    class_name_dict = sample_data.load_class_index_to_name_dict()

    # Load the test data.
    # The provided input images do not have uniform size,
    # so we cannot use batch_size > 1.
    dataset = sample_data.TestDataset()
    dataloader = DataLoader(dataset, batch_size=1)

    # Load pretrained model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet50(weights = ResNet50_Weights.DEFAULT).to(device)
    print(model)

    # Turn off dropout and batch normalization
    model.eval()

    # Get all sequential layers
    sequential_layers = split_resnet(model)

    # Test accuracy.
    print("testing the model with one sample for each class...")
    num_tested = 0
    num_correct = 0
    
    with torch.no_grad():
        for image, label in dataloader:
            # Forward pass.
            pred = model(image.to(device))

            split_pred = image.to(device)
            for i in range(len(sequential_layers)):
                split_pred = sequential_layers[i](split_pred)

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