import io
import itertools
import sys
import time
from PIL import Image
from splitter import *
import torch
from app import *

if __name__ == "__main__":
    image_file_path = sys.argv[1]
    image = Image.open(image_file_path)

    original_model = load_pretrained_resnet()
    module_list = convert_to_module_list(original_model)

    times = []

    combination = itertools.combinations(range(1, 50), 3)

    with torch.no_grad():
        for first, second, third in combination:
            start_time = time.time()
            split_model1 = torch.nn.Sequential(*module_list[0:first]).to(device)
            split_model1.eval()
            input1 = preprocess_image(image).unsqueeze(0).to(device)
            output1 = split_model1(input1).to(device)

            first_check_point_time = time.time()
            split_model2 = torch.nn.Sequential(*module_list[first:second]).to(device)
            split_model2.eval()
            input2 = torch.from_numpy(output1.numpy()).to(device)
            output2 = split_model2(input2).to(device)

            second_check_point_time = time.time()
            split_model3 = torch.nn.Sequential(*module_list[second:third]).to(device)
            split_model3.eval()
            input3 = torch.from_numpy(output2.numpy()).to(device)
            output3 = split_model3(input3).to(device)

            third_check_point_time = time.time()
            split_model4 = torch.nn.Sequential(*module_list[third:50]).to(device)
            split_model4.eval()
            input4 = torch.from_numpy(output3.numpy()).to(device)
            output4 = split_model4(input4).to(device)

            fourth_check_point_time = time.time()
            max_value = max(output4.tolist()[0])

            times.append(
                (
                    (0, first, second, third, 50),
                    (
                        first_check_point_time - start_time,
                        second_check_point_time - first_check_point_time,
                        third_check_point_time - second_check_point_time,
                        fourth_check_point_time - third_check_point_time,
                    ),
                    fourth_check_point_time - start_time,
                )
            )
            print(
                f"""time (0, {first}) : {first_check_point_time - start_time}
time ({first}, {second}) : {second_check_point_time - first_check_point_time}
time ({second}, {third}) : {third_check_point_time - second_check_point_time}
time ({third}, 50) : {fourth_check_point_time - third_check_point_time}
result : {max_value}
"""
            )

    max_throughput = max(times, key=lambda x: sum(x[1]))
    print(
        f"""max throughput
split points : {max_throughput[0]}
time for each section : {max_throughput[1]}
total time : {max_throughput[2]}
"""
    )
