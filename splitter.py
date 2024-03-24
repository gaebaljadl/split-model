import torch
from torchvision.models import ResNet, resnet50, ResNet50_Weights

class ResNetOutputLayer(torch.nn.Module):
    """
    Encapsulates the last part of a ResNet model
    where it uses torch.flatten() function in the middle.

    Since we want to split the whole model into layers,
    the flatten-part had to be grouped into this custom module.
    """

    def __init__(self, avgpool, fc):
        super().__init__()

        self.avgpool = avgpool
        self.fc = fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def load_pretrained_resnet() -> ResNet:
    return resnet50(weights = ResNet50_Weights.DEFAULT)


def convert_to_module_list(resnet_model: ResNet) -> list[torch.nn.Module]:
    """
    Extract modules (i.e. layers) from the ResNet model
    and put them into a list
    """
    layers = []

    layers.append(resnet_model.conv1)
    layers.append(resnet_model.bn1)
    layers.append(resnet_model.relu)
    layers.append(resnet_model.maxpool)

    # These layers are nn.Sequential, so we should iterate over them.
    for layer in resnet_model.layer1:
        layers.append(layer)
    for layer in resnet_model.layer2:
        layers.append(layer)
    for layer in resnet_model.layer3:
        layers.append(layer)
    for layer in resnet_model.layer4:
        layers.append(layer)
    
    # Output layer utilizes torch.flatten(),
    # so it was inevitable to encapsulate
    # the last layers into a custom module.
    layers.append(ResNetOutputLayer(resnet_model.avgpool, resnet_model.fc))

    return layers


def split_module_list(module_list: list[torch.nn.Module], split_layer_index: int) -> tuple[torch.nn.Module, torch.nn.Module]:
    """
    Split given ModuleList into two parts: head and tail.
    module_list[split_layer_index] will be the first layer of the tail.
    """
    head_modules = module_list[:split_layer_index]
    head = torch.nn.Sequential(*head_modules)

    tail_modules = module_list[split_layer_index:]
    tail = torch.nn.Sequential(*tail_modules)

    return head, tail


if __name__ == "__main__":
    # This section will be executed while building docker image
    # so that we can skip downloading pretrained weights everytime.
    model = load_pretrained_resnet()

    # Split the model into head/tail and save them.
    module_list = convert_to_module_list(model)
    split_layer_index = len(module_list) // 2
    head, tail = split_module_list(module_list, split_layer_index)

    torch.save(head, "./head.pth")
    torch.save(tail, "./tail.pth")