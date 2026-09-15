from __future__ import annotations

import hashlib
import importlib.machinery
import json
import shutil
import sys
import types
from pathlib import Path

import torch
from torch import nn

WORKSPACE = Path.cwd()
KARUME = WORKSPACE / "karume"
WASTE = WORKSPACE / "WASTE_APPS"
OUT = WASTE / "Assets" / "anima_vae_encoder"
TMP = WORKSPACE / ".anima-vae-encoder-build"
PUBLISH = TMP / "publish"
BACKUP = WASTE / "Assets" / ".anima_vae_encoder.backup"

sys.path.insert(0, str(KARUME / "tools" / "exporter" / "src"))
sys.path.insert(0, str(KARUME / "tools" / "export-recipes"))

from diffusers import AutoencoderKLQwenImage
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
    QwenImageAttentionBlock,
    QwenImageCausalConv3d,
    QwenImageResample,
    QwenImageRMS_norm,
    QwenImageUpsample,
)

# Karume's generic exporter imports torchvision only to register the deform_conv2d op key.
# This VAE export never executes that op. If torchvision is absent or ABI-mismatched with the
# user's existing torch, register the schema-only op and a tiny import shim instead of replacing torch.
_TORCHVISION_SHIM_LIB = None
try:
    import torchvision as _torchvision  # noqa: F401
except Exception:
    sys.modules.pop("torchvision", None)
    _TORCHVISION_SHIM_LIB = torch.library.Library("torchvision", "FRAGMENT")
    try:
        _TORCHVISION_SHIM_LIB.define(
            "deform_conv2d(Tensor input, Tensor weight, Tensor offset, Tensor mask, Tensor bias, "
            "int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w, "
            "int n_weight_grps, int n_offset_grps, bool use_mask) -> Tensor"
        )
    except RuntimeError as exc:
        if "already" not in str(exc).lower():
            raise
    _torchvision = types.ModuleType("torchvision")
    _torchvision.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None)
    sys.modules["torchvision"] = _torchvision

from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.pipeline import export_to_file
from karume.quantize import round_weights_to_f16
from karume.shards import resolve_shards
from anima import patch

MODEL_ID = "circlestone-labs/Anima-Base-v1.0-Diffusers"
VARIANTS = (
    (512, 512),
    (512, 768),
    (768, 512),
    (576, 1024),
    (1024, 576),
    (768, 768),
    (768, 1024),
    (1024, 768),
    (1024, 1024),
    (832, 1216),
    (1216, 832),
    (768, 1344),
    (1344, 768),
    (640, 1536),
    (1536, 640),
)
VERIFY_VARIANTS = ((512, 512), (512, 768))
SEED = 20260915


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def image_resample_forward(self: nn.Module, x: torch.Tensor, feat_cache=None, feat_idx=None):
    if feat_cache is not None:
        raise NotImplementedError("image-only VAE encoder patch does not accept feat_cache")
    if self.mode not in ("none", "downsample2d", "downsample3d", "upsample2d", "upsample3d"):
        raise NotImplementedError(f"unsupported QwenImageResample mode: {self.mode}")
    # For the first/only image frame, the temporal branch of both *sample3d modes is not executed
    # in diffusers. self.resample is therefore the exact T=1 path.
    return self.resample(x)


def apply_encoder_patch(vae: nn.Module) -> None:
    # Reuse Karume's already-verified rank4 implementations for RMS norm, upsample and attention.
    QwenImageRMS_norm.forward = patch._rms_norm_forward
    QwenImageResample.forward = image_resample_forward
    QwenImageUpsample.forward = patch._upsample_forward
    QwenImageAttentionBlock.forward = patch._attention_block_forward

    slots: list[tuple[nn.Module, str]] = [(vae, "quant_conv")]
    slots += [
        (parent, name)
        for parent in vae.encoder.modules()
        for name, child in parent.named_children()
        if isinstance(child, QwenImageCausalConv3d)
    ]
    for parent, name in slots:
        child = getattr(parent, name)
        if isinstance(child, QwenImageCausalConv3d):
            setattr(parent, name, patch._causal_conv3d_to_conv2d(child))

    for module in vae.encoder.modules():
        if isinstance(module, QwenImageRMS_norm):
            target = (1, module.gamma.numel(), 1, 1)
            if tuple(module.gamma.shape) != target:
                module.gamma = nn.Parameter(module.gamma.detach().reshape(target))


class AnimaVaeEncoder(nn.Module):
    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae
        self.z_dim = int(vae.config.z_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        moments = self.vae.quant_conv(self.vae.encoder(image))
        # DiagonalGaussianDistribution.mode() is the first half of moments.
        return moments[:, : self.z_dim]


def reference_encode(vae: nn.Module, image: torch.Tensor) -> torch.Tensor:
    return vae.encode(image.unsqueeze(2), return_dict=False)[0].mode().squeeze(2)


def verify_equivalence(vae: nn.Module, refs: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]) -> None:
    wrapper = AnimaVaeEncoder(vae).eval()
    with torch.inference_mode():
        for size, (image, expected) in refs.items():
            actual = wrapper(image)
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
            max_abs = float((actual - expected).abs().max())
            print(f"[verify] {size[0]}x{size[1]} max_abs={max_abs:.7g}", flush=True)


def export_variant(wrapper: nn.Module, width: int, height: int, target: Path) -> tuple[Path, ...]:
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / "model.safetensors"
    generator = torch.Generator().manual_seed(SEED + width * 17 + height)
    example = torch.rand((1, 3, height, width), generator=generator, dtype=torch.float32) * 2 - 1
    export_to_file(
        wrapper,
        (example,),
        model_path,
        weight_dtype="f16",
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    )
    shards = resolve_shards(model_path)
    if len(shards) < 2:
        raise RuntimeError(f"{width}x{height}: expected graph + weight shards, got {len(shards)}")
    return shards


def main() -> None:
    if not KARUME.is_dir():
        raise SystemExit(f"Karume checkout not found: {KARUME}")
    if not WASTE.is_dir():
        raise SystemExit(f"WASTE_APPS checkout not found: {WASTE}")

    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)
    PUBLISH.mkdir(parents=True)

    print("[load] Qwen Image VAE", flush=True)
    vae = AutoencoderKLQwenImage.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    vae.use_tiling = False
    vae.use_slicing = False

    print("[quant] round effective VAE weights to f16-representable values", flush=True)
    report = round_weights_to_f16(vae)
    print(f"[quant] {report.describe()}", flush=True)

    refs: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.inference_mode():
        for index, (width, height) in enumerate(VERIFY_VARIANTS):
            generator = torch.Generator().manual_seed(SEED + index)
            image = torch.rand((1, 3, height, width), generator=generator, dtype=torch.float32) * 2 - 1
            print(f"[reference] {width}x{height}", flush=True)
            expected = reference_encode(vae, image)
            refs[(width, height)] = (image, expected)

    print("[patch] exact T=1 image encoder lowering", flush=True)
    apply_encoder_patch(vae)
    wrapper = AnimaVaeEncoder(vae).eval()
    verify_equivalence(vae, refs)
    refs.clear()

    canonical_weights: list[Path] = []
    canonical_hashes: list[str] = []
    manifest_variants: dict[str, dict[str, object]] = {}

    for width, height in VARIANTS:
        key = f"{width}x{height}"
        print(f"[export] {key}", flush=True)
        variant_dir = TMP / key
        shards = export_variant(wrapper, width, height, variant_dir)
        graph_src, *weight_srcs = shards

        graph_name = f"graph-{key}.safetensors"
        shutil.copy2(graph_src, PUBLISH / graph_name)

        if not canonical_weights:
            total = len(weight_srcs)
            for index, source in enumerate(weight_srcs, 1):
                name = f"weights-{index:05d}-of-{total:05d}.safetensors"
                dest = PUBLISH / name
                shutil.copy2(source, dest)
                canonical_weights.append(dest)
                canonical_hashes.append(sha256(dest))
        else:
            if len(weight_srcs) != len(canonical_weights):
                raise RuntimeError(f"{key}: weight shard count changed: {len(weight_srcs)} != {len(canonical_weights)}")
            hashes = [sha256(path) for path in weight_srcs]
            if hashes != canonical_hashes:
                raise RuntimeError(f"{key}: weight shards differ from canonical 512x512 export")

        manifest_variants[key] = {
            "width": width,
            "height": height,
            "shards": [graph_name, *[path.name for path in canonical_weights]],
        }
        shutil.rmtree(variant_dir, ignore_errors=True)

    manifest = {
        "format": "mica-anima-vae-encoder/1",
        "source": MODEL_ID,
        "dtype": "f16",
        "latent_channels": 16,
        "spatial_compression": 8,
        "variants": manifest_variants,
        "weights": [
            {"path": path.name, "size": path.stat().st_size, "sha256": digest}
            for path, digest in zip(canonical_weights, canonical_hashes, strict=True)
        ],
    }
    (PUBLISH / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Publish transactionally: keep the previous working asset until every export/hash/manifest step passed.
    shutil.rmtree(BACKUP, ignore_errors=True)
    try:
        if OUT.exists():
            OUT.replace(BACKUP)
        PUBLISH.replace(OUT)
    except BaseException:
        if not OUT.exists() and BACKUP.exists():
            BACKUP.replace(OUT)
        raise
    shutil.rmtree(BACKUP, ignore_errors=True)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"DONE: {OUT}", flush=True)


if __name__ == "__main__":
    main()
