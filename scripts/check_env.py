import torch, torchvision, yaml, cv2, PIL, numpy
print("torch       =", torch.__version__)
print("torchvision =", torchvision.__version__)
print("cv2         =", cv2.__version__)
print("PIL         =", PIL.__version__)
print("numpy       =", numpy.__version__)
print("cuda        =", torch.cuda.is_available())
