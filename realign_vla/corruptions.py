"""Independent training bank applied to raw HxWx3 uint8 simulator images.

No LIBERO-Plus operators, fog, camera or lighting transformations are used.
Ranges and random-number consumption match the checkpoint-producing source.
"""

import cv2
import numpy as np

SPEC = {
    "gaussian_blur": {"probability": 0.25, "sigma": [0.5, 11.0]},
    "motion_blur": {"probability": 0.25, "length_inclusive": [3, 45], "angle_degrees": [0, 360]},
    "zoom_blur": {"probability": 0.20, "maximum_zoom": [1.05, 1.6], "zoom_samples": 10},
    "glass_blur": {"probability": 0.15, "sigma": [0.5, 3.0], "delta_inclusive": [1, 6]},
    "gaussian_noise": {"probability": 0.15, "std_fraction": [0.01, 0.08]},
}


def gaussian_blur(img, rng):
    s = rng.uniform(0.5, 11.0)
    return cv2.GaussianBlur(img, (0, 0), sigmaX=s, borderType=cv2.BORDER_REFLECT), {"sigma": s}


def motion_blur(img, rng):
    length, angle = int(rng.integers(3, 46)), rng.uniform(0, 360)
    k = np.zeros((length, length), np.float32)
    c = (length - 1) / 2
    dx, dy = np.cos(np.deg2rad(angle)) * c, np.sin(np.deg2rad(angle)) * c
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
    k /= max(k.sum(), 1e-6)
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT), {"length": length, "angle": angle}


def zoom_blur(img, rng):
    zmax = rng.uniform(1.05, 1.6)
    h, w = img.shape[:2]
    acc = img.astype(np.float32)
    zs = np.linspace(1.0, zmax, 10)[1:]
    for z in zs:
        big = cv2.resize(img, (int(round(w * z)), int(round(h * z))), interpolation=cv2.INTER_LINEAR)
        y0, x0 = (big.shape[0] - h) // 2, (big.shape[1] - w) // 2
        acc += big[y0:y0 + h, x0:x0 + w]
    return np.clip(acc / (len(zs) + 1), 0, 255).astype(np.uint8), {"zmax": zmax}


def glass_blur(img, rng):
    s, d = rng.uniform(0.5, 3.0), int(rng.integers(1, 7))
    h, w = img.shape[:2]
    x = cv2.GaussianBlur(img, (0, 0), sigmaX=s)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    gx += rng.integers(-d, d + 1, (h, w)).astype(np.float32)
    gy += rng.integers(-d, d + 1, (h, w)).astype(np.float32)
    x = cv2.remap(x, gx, gy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    return cv2.GaussianBlur(x, (0, 0), sigmaX=s), {"sigma": s, "delta": d}


def gaussian_noise(img, rng):
    s = rng.uniform(0.01, 0.08)
    out = img.astype(np.float32) / 255 + rng.normal(0, s, img.shape)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8), {"std": s}


BANK = {"gaussian_blur": (gaussian_blur, 0.25), "motion_blur": (motion_blur, 0.25), "zoom_blur": (zoom_blur, 0.20),
        "glass_blur": (glass_blur, 0.15), "gaussian_noise": (gaussian_noise, 0.15)}


def random_corruption(img, rng):
    names = list(BANK)
    p = np.array([BANK[n][1] for n in names])
    name = names[rng.choice(len(names), p=p / p.sum())]
    out, params = BANK[name][0](img, rng)
    return out, name, params
