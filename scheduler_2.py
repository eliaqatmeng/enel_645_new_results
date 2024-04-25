import subprocess
import os

script_path1 = os.path.abspath("apr24/vgg16/VGG16_trained.py")
script_path2 = os.path.abspath("apr24/densenet/densenet_trained.py")
script_path3 = os.path.abspath("apr24/resnet50/resnet50.py")
script_path4 = os.path.abspath("apr24/resnet101/resnet101.py")
script_path6 = os.path.abspath("apr24/resnet152/resnet152.py")
script_path7 = os.path.abspath("apr24/vgg16_untrained/VGG16_untrained.py")
script_path8 = os.path.abspath("apr24/densenet_untrained/densenet_untrained.py")
script_path9 = os.path.abspath("apr24/resnet50_untrained/resnet50_untrained.py")
script_path10 = os.path.abspath("apr24/resnet101_untrained/resnet101_untrained.py")
script_path11 = os.path.abspath("apr24/resnet152_untrained/resnet152_untrained.py")

try:
    subprocess.run(["python", script_path1], check=True)
    print("\nVGG16_trained execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("\nVGG16_trained execution failed with error:", e)

try:
    subprocess.run(["python", script_path2], check=True)
    print("\ndensenet_trained execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("\ndensenet_trained execution failed with error:", e)

try:
    subprocess.run(["python", script_path3], check=True)
    print("\nresnet50/resnet50 execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("\nresnet50/resnet50 execution failed with error:", e)

try:
    subprocess.run(["python", script_path4], check=True)
    print("\nresnet101 execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("\nresnet101 execution failed with error:", e)

try:
    subprocess.run(["python", script_path6], check=True)
    print("\nresnet152 execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("\nresnet152 execution failed with error:", e)

try:
    subprocess.run(["python", script_path7], check=True)
    print("\nVGG16_untrained execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("\nVGG16_untrained execution failed with error:", e)

try:
    subprocess.run(["python", script_path8], check=True)
    print("\ndensenet_untrained execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("\ndensenet_untrained execution failed with error:", e)

try:
    subprocess.run(["python", script_path9], check=True)
    print("\nresnet50_untrained execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("\nresnet50_untrained execution failed with error:", e)

try:
    subprocess.run(["python", script_path10], check=True)
    print("\nresnet101_untrained execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("\nresnet101_untrained execution failed with error:", e)

try:
    subprocess.run(["python", script_path11], check=True)
    print("\nresnet152_untrained execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("\nresnet152_untrained execution failed with error:", e)

