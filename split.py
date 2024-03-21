import sample_data
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights


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

    # Test accuracy.
    print("testing the model with one sample for each class...")
    num_tested = 0
    num_correct = 0
    for image, label in dataloader:
        # Forward pass.
        pred = model(image.to(device))

        # Get the class index of prediction with highest probability.
        max_prob_class = torch.argmax(pred)

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