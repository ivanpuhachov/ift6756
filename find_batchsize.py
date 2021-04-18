import torch
import time
import numpy as np
import matplotlib.pyplot as plt

from simple_gan import SimpleGAN
from vector_gan import BezierGAN, SimpleGANBezier


def approximate_forward(batch_size, model):
    model.train()
    # with torch.no_grad():
    #     fake_data = model.generator.generate_batch(batch_size=batch_size).detach()
    fake_data = model.generator.generate_batch(batch_size=batch_size).detach()
    decision_fake = model.discriminator(fake_data)
    loss_fake = torch.mean(decision_fake)
    loss = loss_fake
    loss.backward()
    return loss.item()

def measure_time(model, batches_to_try = [2,10]):
    results = list()
    for bs in batches_to_try:
        start = time.time()
        for i in range(20):
            l = approximate_forward(bs, model)
        end = time.time()
        results.append(end-start)
        print(f"batch {bs}: {(end-start)/bs:.2f}")
    return results


if __name__=="__main__":
    model = BezierGAN().cuda()
    batches_to_try = np.array([5,10,20, 25, 30, 35, 40, 45, 50,60,80,100,120,140])
    # batches_to_try = np.array([5,10,20,40,60,80,])
    results = np.array(measure_time(model, batches_to_try=batches_to_try))
    plt.scatter(batches_to_try, results/batches_to_try)
    plt.xticks(batches_to_try)
    plt.show()
