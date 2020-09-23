import torch
import torch.optim as optim
import torchvision.models as models
import math

#for visualization
import matplotlib.pyplot as plt


def cosine_annealing(optimizer, start_lr,cur_steps, num_cycle):
    t_cur = cur_steps % num_cycle
    lr = 0.5 * start_lr * (math.cos(math.pi * t_cur / num_cycle) + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def main():

    model = models.vgg11_bn(pretrained=True)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    lr = []
    for e in range(100):
        lr.append(cosine_annealing(optimizer, 0.1, e, 100))
    plt.plot(range(100), lr)        
    plt.show()

if __name__ == '__main__':
    main()