import PIL
import os
import ast
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset


class TestDataset(Dataset):
    """
    Provide a pytorch dataset from sample images downloaded from
    https://github.com/EliSchwartz/imagenet-sample-images.git.

    git clone must be performed prior to creating this dataset object.
    """

    def __init__(self):
        # Get the list of all image files downloaded from the repository.
        # File extension check is required to exclude
        # non-image files such as README.md.
        directory = "./imagenet-sample-images"
        extension = ".JPEG"
        self.file_names = list(filter(lambda name : name.endswith(extension), os.listdir(directory)))

        # We must have 1000 images in the directory.
        assert(len(self.file_names) == 1000)

        # The name of each sample files represent the class they belong.
        self.labels = [parse_label(name) for name in self.file_names]

        # Load images using PIL
        to_tensor = ToTensor()
        self.images = [to_tensor(PIL.Image.open("{}/{}".format(directory, name))) for name in self.file_names]

        # Convert grayscale to RGB by duplicating color value 3 times
        for i in range(len(self.images)):
            is_grayscale = self.images[i].size()[0] == 1
            if is_grayscale:
                self.images[i] = self.images[i].repeat(3, 1, 1)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def parse_label(file_name: str) -> str:
    """
    Get the label name of ImageNet sample data downloaded
    from https://github.com/EliSchwartz/imagenet-sample-images.git.

    1. remove suffix ".JPEG"
    2. remove prefix "nXXXX_"
    3. convert "_" to " "

    ex) "n03085013_computer_keyboard.JPEG" => "computer keyboard"
    """
    tokens = file_name.removesuffix('.JPEG').split('_')
    return ' '.join(tokens[1:])


def load_class_index_to_name_dict() -> dict:
    """
    Parse the content downloaded from https://gist.github.com/942d3a0ac09ec9e5eb3a.git
    to get the mapping from class index to class name.
    """
    # The file is formatted like a dict (ex. {0: 'asdf', 1: 'qwerty', ...}),
    # so we can use ast.literal_eval() to parse it directly.
    with open("./942d3a0ac09ec9e5eb3a/imagenet1000_clsidx_to_labels.txt", "r") as file:
        return ast.literal_eval(file.read())
