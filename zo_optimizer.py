"""
zo_optimizer.py — Zero-order optimizer skeleton.
"""

from __future__ import annotations
import math
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader


class ZeroOrderOptimizer:
    """SPSA optimizer: sklearn LR warm-start + SPSA bias tuning.

    Hyperparameters:
        lr              Peak Adam lr for bias. Decays cosine to lr_min.
        lr_min          Minimum Adam lr (end of cosine schedule).
        eps             SPSA perturbation magnitude.
        beta1/beta2     Adam moment decay.
        adam_eps        Adam numerical stabiliser.
        n_spsa_samples  SPSA estimates averaged per step.
        lr_C            sklearn LR regularisation (C=1/lambda).
        data_dir        Path to CIFAR100 root (downloaded automatically).
        n_batches       Total optimizer steps (for cosine decay schedule).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        lr_min: float = 1e-5,
        eps: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        adam_eps: float = 1e-8,
        n_spsa_samples: int = 500,
        lr_C: float = 0.316,
        data_dir: str = "./data",
        n_batches: int = 128,
    ) -> None:
        self.model = model
        self.lr = lr
        self.lr_min = lr_min
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.adam_eps = adam_eps
        self.n_spsa_samples = n_spsa_samples
        self.lr_C = lr_C
        self.data_dir = data_dir
        self.n_batches = n_batches

        self._step_count: int = 0
        self._m_bias: torch.Tensor | None = None
        self._v_bias: torch.Tensor | None = None

        self.layer_names: list[str] = ["fc.bias"]

        try:
            self._device = next(model.parameters()).device
        except StopIteration:
            self._device = torch.device("cpu")

        # sklearn LR warm-start
        self._lr_warmstart()

    # sklearn LogisticRegression warm-start

    def _lr_warmstart(self) -> None:
        """Fit a logistic regression on backbone features and set fc.weight/bias."""
        print("[ZO] Fitting sklearn LogisticRegression on backbone features ...")

        _CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
        _CIFAR100_STD = (0.2675, 0.2565, 0.2761)
        transform = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
        ])

        try:
            train_dataset = datasets.CIFAR100(
                root=self.data_dir, train=True, download=True, transform=transform
            )
        except Exception as exc:
            print(f"[ZO] Warning: CIFAR100 load failed: {exc}. Using centroid fallback.")
            self._centroid_fallback()
            return

        loader = DataLoader(
            train_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        self.model.eval()
        self.model.to(self._device)

        all_features, all_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self._device)
                feats = self._extract_features(images)
                all_features.append(feats.cpu())
                all_labels.append(labels)

        X = torch.cat(all_features).numpy()
        y = torch.cat(all_labels).numpy()

        print(f"[ZO] Features extracted: {X.shape}. Fitting LR (C={self.lr_C}) ...")

        norms = np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-8)
        X_norm = X / norms

        lr_clf = LogisticRegression(
            C=self.lr_C,
            max_iter=1000,
            solver="lbfgs",
            multi_class="multinomial",
            n_jobs=-1,
            verbose=0,
            tol=1e-4,
        )
        lr_clf.fit(X_norm, y)

        print(f"[ZO] LR fit done. Train accuracy: {lr_clf.score(X_norm, y)*100:.2f}%")

        W = torch.tensor(lr_clf.coef_, dtype=torch.float32)
        b = torch.tensor(lr_clf.intercept_, dtype=torch.float32)

        with torch.no_grad():
            self.model.fc.weight.copy_(W)
            self.model.fc.bias.copy_(b)

        print("[ZO] fc.weight and fc.bias set from LR coefficients.")

    def _centroid_fallback(self) -> None:
        """Fallback if sklearn or data are unavailable: use L2-normalized centroids."""
        print("[ZO] Using centroid fallback init.")
        _CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
        _CIFAR100_STD = (0.2675, 0.2565, 0.2761)
        transform = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
        ])
        try:
            ds = datasets.CIFAR100(
                root=self.data_dir, train=True, download=True, transform=transform
            )
            loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
            n_classes, feature_dim = 100, self.model.fc.in_features
            class_sums = torch.zeros(n_classes, feature_dim, device=self._device)
            class_counts = torch.zeros(n_classes, device=self._device)
            self.model.eval()
            with torch.no_grad():
                for images, labels in loader:
                    images = images.to(self._device)
                    labels = labels.to(self._device)
                    feats = self._extract_features(images)
                    class_sums.scatter_add_(
                        0, labels.unsqueeze(1).expand(-1, feature_dim), feats
                    )
                    class_counts.scatter_add_(
                        0, labels, torch.ones(labels.size(0), device=self._device)
                    )
            centroids = F.normalize(
                class_sums / class_counts.unsqueeze(1).clamp(min=1), dim=1
            )
            with torch.no_grad():
                self.model.fc.weight.copy_(centroids)
                self.model.fc.bias.zero_()
        except Exception:
            nn.init.orthogonal_(self.model.fc.weight, gain=1.0)
            nn.init.zeros_(self.model.fc.bias)

    # Backbone feature extraction

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """ResNet18 backbone (all layers before fc)."""
        x = self.model.conv1(images)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        return torch.flatten(x, 1)

    # SPSA gradient estimation (bias only)

    def _estimate_bias_grad(
        self,
        loss_fn: Callable[[], float],
        cached_features: torch.Tensor | None,
        cached_labels: torch.Tensor | None,
    ) -> torch.Tensor:
        """Average SPSA gradient estimate for fc.bias."""
        criterion = nn.CrossEntropyLoss()
        fc_bias = self.model.fc.bias
        acc = torch.zeros_like(fc_bias.data)

        with torch.no_grad():
            for _ in range(self.n_spsa_samples):
                delta = (
                    torch.randint(
                        0, 2, fc_bias.shape, device=self._device, dtype=fc_bias.dtype
                    )
                    * 2.0
                    - 1.0
                )

                fc_bias.data.add_(self.eps * delta)
                f_plus = (
                    criterion(self.model.fc(cached_features), cached_labels).item()
                    if cached_features is not None
                    else loss_fn()
                )

                fc_bias.data.sub_(2.0 * self.eps * delta)
                f_minus = (
                    criterion(self.model.fc(cached_features), cached_labels).item()
                    if cached_features is not None
                    else loss_fn()
                )

                fc_bias.data.add_(self.eps * delta)

                g_hat = (f_plus - f_minus) / (2.0 * self.eps)
                acc.add_(g_hat * delta)

        return acc / self.n_spsa_samples

    # Cosine lr schedule

    def _current_lr(self) -> float:
        """Cosine annealing: lr decays from lr to lr_min over n_batches."""
        t = self._step_count - 1
        T = max(self.n_batches - 1, 1)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * t / T))
        return self.lr_min + (self.lr - self.lr_min) * cos_factor

    # Adam update (bias only)

    def _update_bias(self, grad: torch.Tensor) -> None:
        """Adam update for fc.bias with cosine lr decay."""
        fc_bias = self.model.fc.bias
        t = self._step_count

        if self._m_bias is None:
            self._m_bias = torch.zeros_like(fc_bias.data)
            self._v_bias = torch.zeros_like(fc_bias.data)

        lr_t = self._current_lr()
        bc1 = 1.0 - self.beta1 ** t
        bc2 = 1.0 - self.beta2 ** t

        with torch.no_grad():
            self._m_bias.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            self._v_bias.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
            m_hat = self._m_bias / bc1
            v_hat = self._v_bias / bc2
            fc_bias.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.adam_eps), value=-lr_t)

    # Public API

    def step(self, loss_fn: Callable[[], float]) -> float:
        """One ZO optimisation step (SPSA on fc.bias with cosine lr)."""
        self._step_count += 1

        cached_features = None
        cached_labels = None

        if (
            hasattr(loss_fn, "__defaults__")
            and loss_fn.__defaults__ is not None
            and len(loss_fn.__defaults__) >= 2
        ):
            images_tensor = loss_fn.__defaults__[0]
            labels_tensor = loss_fn.__defaults__[1]
            if isinstance(images_tensor, torch.Tensor):
                self.model.eval()
                with torch.no_grad():
                    feats = self._extract_features(images_tensor.to(self._device))
                    norms = feats.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    cached_features = feats / norms
                cached_labels = labels_tensor.to(self._device)

        with torch.no_grad():
            loss_before = float(loss_fn())

        grad_bias = self._estimate_bias_grad(loss_fn, cached_features, cached_labels)
        self._update_bias(grad_bias)

        return float(loss_before)
