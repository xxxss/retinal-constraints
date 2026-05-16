"""
Test individual retinal layer contributions.

Start with the simplest: photoreceptor log compression.
Then stack layers to see cumulative effect.

  1. baseline:          ResNet18
  2. log_only:          log compression → ResNet18
  3. dog_only:          adaptive DoG + aux supervision → ResNet18 (current best)
  4. log_then_dog:      log compression → adaptive DoG + aux → ResNet18
"""

import json, os, time
import matplotlib
matplotlib.use("Agg")
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


class PhotoreceptorLog(nn.Module):
    """Photoreceptor-inspired log compression.

    Biological basis: photoreceptor output is approximately log(light intensity).
    This compresses dynamic range — moonlight (0.01 cd/m2) and sunlight (100,000 cd/m2)
    both produce usable signals.

    For images in [0, 1]:  output = log(1 + scale * x) / log(1 + scale)
    This maps [0, 1] → [0, 1] with a compressive nonlinearity.
    """
    def __init__(self, scale=10.0):
        super().__init__()
        self.scale = scale
        self.norm = torch.log1p(torch.tensor(scale))

    def forward(self, x):
        # Clamp to avoid log of negative values from corruptions
        return torch.log1p(x.clamp(min=0) * self.scale) / self.norm


class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(weights=None, num_classes=10)
    def forward(self, x):
        return self.net(x)


class LogModel(nn.Module):
    """Just log compression, no DoG."""
    def __init__(self):
        super().__init__()
        self.photo = PhotoreceptorLog()
        self.net = models.resnet18(weights=None, num_classes=10)
    def forward(self, x):
        return self.net(self.photo(x))


class DogModel(nn.Module):
    """Adaptive DoG + aux supervision (our current best)."""
    def __init__(self):
        super().__init__()
        self.dog = AdaptiveDoG(in_channels=3, kernel_size=9)
        self.adapter = nn.Conv2d(6, 3, 1, bias=False)
        nn.init.kaiming_normal_(self.adapter.weight)
        self.net = models.resnet18(weights=None, num_classes=10)
    def forward(self, x):
        return self.net(self.adapter(self.dog(x)))
    def get_noise_level_tensor(self, x):
        return self.dog._estimate_noise(x)


class LogDogModel(nn.Module):
    """Log compression → Adaptive DoG → ResNet18.
    Mimics: photoreceptor → horizontal/bipolar → ganglion cell."""
    def __init__(self):
        super().__init__()
        self.photo = PhotoreceptorLog()
        self.dog = AdaptiveDoG(in_channels=3, kernel_size=9)
        self.adapter = nn.Conv2d(6, 3, 1, bias=False)
        nn.init.kaiming_normal_(self.adapter.weight)
        self.net = models.resnet18(weights=None, num_classes=10)
    def forward(self, x):
        x = self.photo(x)
        x = self.dog(x)
        x = self.adapter(x)
        return self.net(x)
    def get_noise_level_tensor(self, x):
        return self.dog._estimate_noise(self.photo(x))


def get_cifar10(batch_size=128):
    t1 = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor()])
    t2 = transforms.Compose([transforms.ToTensor()])
    tr = datasets.CIFAR10("./data", train=True, download=True, transform=t1)
    te = datasets.CIFAR10("./data", train=False, download=True, transform=t2)
    return (torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            torch.utils.data.DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True))


def train(model, epochs, device, noise_aug=False, sup_est=False):
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
            is_noisy = False
            if noise_aug and torch.rand(1).item() > 0.5:
                c = augs[torch.randint(len(augs), (1,)).item()]
                imgs = apply_corruption(imgs, c, torch.randint(1, 4, (1,)).item())
                is_noisy = True
            opt.zero_grad()
            out = model(imgs)
            loss = cls_crit(out, labs)
            if sup_est and hasattr(model, 'get_noise_level_tensor'):
                np_ = model.get_noise_level_tensor(imgs)
                nt = torch.full_like(np_, 1.0 if is_noisy else 0.0)
                loss = loss + 0.5 * bce_crit(np_, nt)
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
    elif mode == "log_only":
        model = LogModel()
    elif mode == "dog_only":
        model = DogModel()
        noise_aug = True; sup_est = True
    elif mode == "log_then_dog":
        model = LogDogModel()
        noise_aug = True; sup_est = True
    else:
        raise ValueError(mode)

    t0 = time.time()
    model, tst = train(model, epochs, device, noise_aug=noise_aug, sup_est=sup_est)
    print("  Training time: {:.0f}s".format(time.time() - t0))

    cl = eval_clean(model, tst, device)
    print("  Clean accuracy: {:.4f}".format(cl))

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

    modes = ["baseline", "log_only", "dog_only", "log_then_dog"]
    R = {}
    for m in modes:
        R[m] = run(m, 50, device)

    sep = "=" * 60
    print("\n" + sep)
    print("RETINA LAYERS SUMMARY")
    print(sep)
    print("{:<16} | {:>7} | {:>9} | {:>7}".format("Mode", "Clean", "Corrupted", "Drop"))
    print("-" * 16 + "-+-" + "-" * 7 + "-+-" + "-" * 9 + "-+-" + "-" * 7)
    for m in modes:
        c = R[m]["clean"] * 100
        cr = R[m]["corruptions"]["mean"] * 100
        print("{:<16} | {:>6.1f}% | {:>8.1f}% | {:>6.1f}%".format(m, c, cr, c - cr))

    # Per-corruption breakdown for the two best models
    print("\n" + sep)
    print("PER-CORRUPTION: dog_only vs log_then_dog")
    print(sep)
    print("{:<20} | {:>10} | {:>12} | {:>7}".format("Corruption", "dog_only", "log+dog", "delta"))
    print("-" * 20 + "-+-" + "-" * 10 + "-+-" + "-" * 12 + "-+-" + "-" * 7)
    for c in CORRUPTIONS:
        d = R["dog_only"]["corruptions"][c] * 100
        ld = R["log_then_dog"]["corruptions"][c] * 100
        print("{:<20} | {:>9.1f}% | {:>11.1f}% | {:>+6.1f}%".format(c, d, ld, ld - d))
