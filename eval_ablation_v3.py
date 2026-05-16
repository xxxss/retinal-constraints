"""
Ablation v3: add auxiliary supervision for the noise estimator.

The noise estimator in v1/v2 couldn't distinguish clean from noisy inputs
because it only received gradients from classification loss, which doesn't
care about filter choice.

Fix: add a binary cross-entropy loss that directly supervises the noise
estimator: "this batch IS corrupted (1) / is NOT corrupted (0)".

This is analogous to how retinal adaptation is hardwired — it's not learned
from task performance, but directly driven by photochemical responses to
light level.

Compares:
  1. baseline:             no DoG, no noise aug
  2. baseline_noiseaug:    no DoG, noise aug
  3. adaptive_dog:         adaptive DoG + noise aug (NO aux loss, same as v2)
  4. adaptive_dog_supervised: adaptive DoG + noise aug + aux loss on estimator (NEW)
"""

import json, os, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from dog_layer_adaptive import AdaptiveDoG

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "gaussian_blur", "brightness", "contrast", "fog", "jpeg",
]

def apply_corruption(images, corruption_type, severity=3):
    s = severity
    if corruption_type == "gaussian_noise":
        return images + torch.randn_like(images) * [.02,.04,.06,.08,.12][s-1]
    elif corruption_type == "shot_noise":
        return images + torch.randn_like(images) * [.015,.03,.05,.07,.1][s-1]
    elif corruption_type == "impulse_noise":
        prob = [.01,.03,.06,.1,.15][s-1]
        mask = torch.rand_like(images[:,:1,:,:]); out = images.clone()
        out[mask.expand_as(out) < prob/2] = 0.0
        out[(mask.expand_as(out) >= prob/2) & (mask.expand_as(out) < prob)] = 1.0
        return out
    elif corruption_type == "gaussian_blur":
        sigma = [.3,.5,.8,1.2,1.8][s-1]; ks = max(3, int(2*np.ceil(2*sigma)+1))
        if ks % 2 == 0: ks += 1
        return transforms.GaussianBlur(ks, sigma)(images)
    elif corruption_type == "brightness":
        return (images + [.05,.1,.15,.2,.3][s-1]).clamp(0, 1)
    elif corruption_type == "contrast":
        f = [.8,.6,.4,.25,.1][s-1]; m = images.mean(dim=(-2,-1), keepdim=True)
        return ((images - m) * f + m).clamp(0, 1)
    elif corruption_type == "fog":
        fl = [.15,.25,.4,.55,.7][s-1]; return (images * (1 - fl) + fl).clamp(0, 1)
    elif corruption_type == "jpeg":
        return (images + torch.randn_like(images) * [.02,.04,.06,.1,.15][s-1]).clamp(0, 1)
    return images


class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(weights=None, num_classes=10)
    def forward(self, x):
        return self.net(x)


class AdaptiveDoGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dog = AdaptiveDoG(in_channels=3, kernel_size=9)
        self.adapter = nn.Conv2d(6, 3, 1, bias=False)
        nn.init.kaiming_normal_(self.adapter.weight)
        self.net = models.resnet18(weights=None, num_classes=10)

    def forward(self, x):
        return self.net(self.adapter(self.dog(x)))

    def get_noise_level(self, x):
        return self.dog.get_noise_level(x)

    def get_noise_level_tensor(self, x):
        """Return noise level as tensor (for loss computation)."""
        return self.dog._estimate_noise(x)  # (B, 1)


def get_cifar10(batch_size=128):
    t1 = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor()])
    t2 = transforms.Compose([transforms.ToTensor()])
    tr = datasets.CIFAR10("./data", train=True, download=True, transform=t1)
    te = datasets.CIFAR10("./data", train=False, download=True, transform=t2)
    return (torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            torch.utils.data.DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True))


def train(model, epochs, device, noise_aug=False, supervise_estimator=False):
    model = model.to(device)
    trn, tst = get_cifar10()
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    cls_crit = nn.CrossEntropyLoss()
    bce_crit = nn.BCELoss()
    augs = ["gaussian_noise", "shot_noise", "gaussian_blur", "contrast", "fog"]

    for ep in range(epochs):
        model.train()
        cor = tot = 0
        for imgs, labs in trn:
            imgs, labs = imgs.to(device), labs.to(device)

            # Decide whether to corrupt this batch
            is_noisy = False
            if noise_aug and torch.rand(1).item() > 0.5:
                c = augs[torch.randint(len(augs), (1,)).item()]
                imgs = apply_corruption(imgs, c, torch.randint(1, 4, (1,)).item())
                is_noisy = True

            opt.zero_grad()
            out = model(imgs)
            loss = cls_crit(out, labs)

            # Auxiliary loss: supervise noise estimator directly
            if supervise_estimator and hasattr(model, "get_noise_level_tensor"):
                noise_pred = model.get_noise_level_tensor(imgs)  # (B, 1)
                noise_target = torch.full_like(noise_pred, 1.0 if is_noisy else 0.0)
                aux_loss = bce_crit(noise_pred, noise_target)
                loss = loss + 0.5 * aux_loss  # weight the aux loss

            loss.backward()
            opt.step()
            cor += (out.argmax(1) == labs).sum().item()
            tot += imgs.size(0)
        sched.step()

        if (ep + 1) % 10 == 0 or ep == 0:
            ta = eval_clean(model, tst, device)
            print("  Epoch {:3d}/{} | train={:.4f} test={:.4f}".format(ep+1, epochs, cor/tot, ta))
    return model, tst


@torch.no_grad()
def eval_clean(model, loader, device):
    model.eval()
    c = t = 0
    for imgs, labs in loader:
        imgs, labs = imgs.to(device), labs.to(device)
        c += (model(imgs).argmax(1) == labs).sum().item()
        t += imgs.size(0)
    return c / t


@torch.no_grad()
def eval_corrupt(model, loader, device, sev=3):
    model.eval()
    res = {}
    for corr in CORRUPTIONS:
        c = t = 0
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            c += (model(apply_corruption(imgs, corr, sev)).argmax(1) == labs).sum().item()
            t += imgs.size(0)
        res[corr] = c / t
    res["mean"] = np.mean([res[c] for c in CORRUPTIONS])
    return res


def run(mode, epochs, device):
    sep = "=" * 60
    print("\n" + sep)
    print("Mode: " + mode)
    print(sep)

    noise_aug = False
    sup_est = False

    if mode == "baseline":
        model = BaselineModel()
    elif mode == "baseline_noiseaug":
        model = BaselineModel()
        noise_aug = True
    elif mode == "adaptive_dog":
        model = AdaptiveDoGModel()
        noise_aug = True
    elif mode == "adaptive_dog_supervised":
        model = AdaptiveDoGModel()
        noise_aug = True
        sup_est = True
    else:
        raise ValueError(mode)

    t0 = time.time()
    model, tst = train(model, epochs, device, noise_aug=noise_aug, supervise_estimator=sup_est)
    print("  Training time: {:.0f}s".format(time.time() - t0))

    cl = eval_clean(model, tst, device)
    print("  Clean accuracy: {:.4f}".format(cl))

    if hasattr(model, "get_noise_level"):
        sb = next(iter(tst))[0][:16].to(device)
        cn = model.get_noise_level(sb)
        nn_ = model.get_noise_level(apply_corruption(sb, "gaussian_noise", 4))
        print("  Noise est -- clean:{:.3f} noisy:{:.3f} (delta:{:.3f})".format(
            cn.mean(), nn_.mean(), nn_.mean() - cn.mean()))

    cr = eval_corrupt(model, tst, device)
    print("")
    print("  {:<20} | {:>10}".format("Corruption", "Accuracy"))
    print("  " + "-" * 20 + "-+-" + "-" * 10)
    for c in CORRUPTIONS:
        print("  {:<20} | {:>9.4f}".format(c, cr[c]))
    print("  " + "-" * 20 + "-+-" + "-" * 10)
    print("  {:<20} | {:>9.4f}".format("MEAN", cr["mean"]))
    print("  {:<20} | {:>9.4f}".format("CLEAN", cl))

    return {"clean": cl, "corruptions": cr}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: " + str(device))

    modes = ["baseline", "baseline_noiseaug", "adaptive_dog", "adaptive_dog_supervised"]
    R = {}
    for m in modes:
        R[m] = run(m, 50, device)

    sep = "=" * 60
    print("\n" + sep)
    print("ABLATION v3 SUMMARY")
    print(sep)
    print("{:<25} | {:>8} | {:>10} | {:>8}".format("Mode", "Clean", "Corrupted", "Drop"))
    print("-" * 25 + "-+-" + "-" * 8 + "-+-" + "-" * 10 + "-+-" + "-" * 8)
    for m in modes:
        c = R[m]["clean"] * 100
        cr = R[m]["corruptions"]["mean"] * 100
        print("{:<25} | {:>7.1f}% | {:>9.1f}% | {:>7.1f}%".format(m, c, cr, c - cr))
