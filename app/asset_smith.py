import argparse
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import forge_shape
import matte_ops
import view_runner_sd
import view_runner_sdxl
from viewforge.pipelines.pipeline_texture import ModProcessConfig, TexturePipeline
from viewforge.utils import make_image_grid


def _join_repo(owner: str, name_parts: Tuple[str, ...]) -> str:
    return f"{owner}/" + "".join(name_parts)


def _join_name(*parts: str) -> str:
    return "".join(parts)


@dataclass(frozen=True)
class EngineSpec:
    shape_repo: str
    shape_subfolder: str
    appearance_repo: str
    appearance_variant: str
    base_model: str
    vae_model: Optional[str]
    height: int
    width: int
    uv_size: int
    adapter_weight: str


def _build_spec(appearance_variant: str) -> EngineSpec:
    shape_repo = _join_repo("tencent", ("Hun", "yuan", "3D", "-2"))
    if appearance_variant == "sd21":
        return EngineSpec(
            shape_repo=shape_repo,
            shape_subfolder=_join_name("hun", "yuan", "3d", "-dit-v2-0"),
            appearance_repo=_join_repo("huanngzh", ("m", "v", "-", "ad", "ap", "ter")),
            appearance_variant=appearance_variant,
            base_model="stabilityai/stable-diffusion-2-1-base",
            vae_model=None,
            height=512,
            width=512,
            uv_size=2048,
            adapter_weight=_join_name("mv", "adapter", "_ig2mv_", "sd21", ".safetensors"),
        )

    return EngineSpec(
        shape_repo=shape_repo,
        shape_subfolder=_join_name("hun", "yuan", "3d", "-dit-v2-0"),
        appearance_repo=_join_repo("huanngzh", ("m", "v", "-", "ad", "ap", "ter")),
        appearance_variant="sdxl",
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        vae_model="madebyollin/sdxl-vae-fp16-fix",
        height=768,
        width=768,
        uv_size=4096,
        adapter_weight=_join_name("mv", "adapter", "_ig2mv_", "sdxl", ".safetensors"),
    )


def _prepare_runtime_layout(base_root: Path):
    weights_dir = base_root / "weights"
    layout = {
        "weights": weights_dir,
        "shape_store": weights_dir / "shape_store",
        "base_models": weights_dir / "base_models",
        "adapters": weights_dir / "adapters",
        "hf_home": weights_dir / "hf_home",
        "checkpoints": weights_dir / "checkpoints",
        "u2net_home": weights_dir / "u2net",
        "torch_extensions": weights_dir / "torch_extensions",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("ASSET_SMITH_MODELS", str(layout["shape_store"]))
    os.environ.setdefault("HF_HOME", str(layout["hf_home"]))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(layout["hf_home"] / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(layout["hf_home"] / "transformers"))
    os.environ.setdefault("U2NET_HOME", str(layout["u2net_home"]))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(layout["torch_extensions"]))

    venv_bin = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")
    os.environ["PATH"] = str(venv_bin) + os.pathsep + os.environ.get("PATH", "")
    return layout


def _prefer_local_model(local_path: Path, fallback: Optional[str]) -> Optional[str]:
    return str(local_path) if local_path.exists() else fallback


def _resolve_model_sources(base_root: Path, spec: EngineSpec):
    layout = _prepare_runtime_layout(base_root)
    variant_tag = "sd21" if spec.appearance_variant == "sd21" else "sdxl"
    resolved = {
        "shape_repo": spec.shape_repo,
        "base_model": _prefer_local_model(
            layout["base_models"] / f"base_model_{variant_tag}",
            spec.base_model,
        ),
        "vae_model": _prefer_local_model(
            layout["base_models"] / f"vae_model_{variant_tag}",
            spec.vae_model,
        ) if spec.vae_model is not None else None,
        "adapter_repo": _prefer_local_model(
            layout["adapters"] / "ms_adapter",
            spec.appearance_repo,
        ),
        "layout": layout,
    }
    return resolved


def _resolve_checkpoint(
    explicit_path: Optional[str],
    base_root: Path,
    default_name: str,
) -> Optional[str]:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    candidates.append(base_root / "weights" / "checkpoints" / default_name)
    candidates.append(base_root / "checkpoints" / default_name)
    candidates.append(base_root / "viewforge" / "checkpoints" / default_name)

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _load_subject(image_path: Path, background_cleaner, keep_background: bool) -> Image.Image:
    image = Image.open(image_path)
    has_alpha = "A" in image.getbands()
    if has_alpha:
        return image.convert("RGBA")

    image = image.convert("RGB")
    if keep_background:
        return image.convert("RGBA")

    return background_cleaner(image).convert("RGBA")


def _pick_generator(seed: int):
    return None if seed < 0 else torch.manual_seed(seed)


def _ensure_output_path(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".glb":
        raise ValueError("The output path must end with '.glb'.")


def _require_ckpt(path_value: Optional[str], label: str):
    if path_value is None:
        raise FileNotFoundError(
            f"Missing required {label} checkpoint. Pass it explicitly with the matching CLI argument."
        )


def build_parser():
    parser = argparse.ArgumentParser(description="Single-image asset builder")
    parser.add_argument("--image", type=str, required=True, help="Reference image path")
    parser.add_argument("--output", type=str, required=True, help="Final .glb path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--prompt", type=str, default="high quality")
    parser.add_argument("--negative_prompt", type=str, default="watermark, ugly, deformed, noisy, blurry, low contrast")
    parser.add_argument("--appearance_variant", type=str, default="sdxl", choices=["sdxl", "sd21"])
    parser.add_argument("--shape_steps", type=int, default=50)
    parser.add_argument("--shape_octree_resolution", type=int, default=380)
    parser.add_argument("--shape_num_chunks", type=int, default=20000)
    parser.add_argument("--appearance_steps", type=int, default=50)
    parser.add_argument("--appearance_guidance_scale", type=float, default=3.0)
    parser.add_argument("--reference_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--keep_background", action="store_true")
    parser.add_argument("--preprocess_mesh", action="store_true")
    parser.add_argument("--disable_upscaler", action="store_true")
    parser.add_argument("--inpaint_mode", type=str, default="view", choices=["view", "uv", "none"])
    parser.add_argument("--upscaler_ckpt", type=str, default=None)
    parser.add_argument("--inpainter_ckpt", type=str, default=None)
    parser.add_argument("--work_dir", type=str, default=None, help="Optional directory for temporary files")
    parser.add_argument("--keep_intermediate", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    output_path = Path(args.output).expanduser().resolve()
    image_path = Path(args.image).expanduser().resolve()
    _ensure_output_path(output_path)

    spec = _build_spec(args.appearance_variant)
    resolved = _resolve_model_sources(project_root, spec)

    background_cleaner = getattr(matte_ops, _join_name("Background", "Remover"))()
    subject = _load_subject(image_path, background_cleaner, keep_background=args.keep_background)

    shape_cls = getattr(forge_shape, _join_name("Hun", "yuan", "3D", "DiT", "Flow", "Matching", "Pipeline"))
    shape_engine = shape_cls.from_pretrained(
        resolved["shape_repo"],
        subfolder=spec.shape_subfolder,
        variant="fp16",
        device=args.device,
        dtype=torch.float16,
    )

    proxy_generator = _pick_generator(args.seed)
    proxy_mesh = shape_engine(
        image=subject,
        num_inference_steps=args.shape_steps,
        octree_resolution=args.shape_octree_resolution,
        num_chunks=args.shape_num_chunks,
        generator=proxy_generator,
        output_type="trimesh",
    )[0]

    if args.work_dir:
        work_root = Path(args.work_dir).expanduser().resolve()
        work_root.mkdir(parents=True, exist_ok=True)
    else:
        work_root = output_path.parent

    temp_context = tempfile.TemporaryDirectory(dir=str(work_root))
    try:
        temp_dir = Path(temp_context.name)
        mesh_path = temp_dir / "proxy_surface.glb"
        packed_view_path = temp_dir / "packed_views.png"
        proxy_mesh.export(mesh_path)
        # Keep multiview synthesis and texture reprojection in the same mesh frame.
        mesh_front_x = False

        vision_pkg = view_runner_sd if spec.appearance_variant == "sd21" else view_runner_sdxl
        pipe = vision_pkg.prepare_pipeline(
            base_model=resolved["base_model"],
            vae_model=resolved["vae_model"],
            unet_model=None,
            lora_model=None,
            adapter_path=resolved["adapter_repo"],
            scheduler=None,
            num_views=6,
            device=args.device,
            dtype=torch.float16,
        )

        images, _, _, _ = vision_pkg.run_pipeline(
            pipe,
            mesh_path=str(mesh_path),
            num_views=6,
            text=args.prompt,
            image=subject,
            height=spec.height,
            width=spec.width,
            num_inference_steps=args.appearance_steps,
            guidance_scale=args.appearance_guidance_scale,
            seed=args.seed,
            remove_bg_fn=None,
            reference_conditioning_scale=args.reference_conditioning_scale,
            negative_prompt=args.negative_prompt,
            lora_scale=1.0,
            front_x=mesh_front_x,
            device=args.device,
        )
        make_image_grid(images, rows=1).save(packed_view_path)

        use_upscaler = not args.disable_upscaler
        upscaler_ckpt = _resolve_checkpoint(
            args.upscaler_ckpt,
            project_root,
            "RealESRGAN_x2plus.pth",
        )
        inpainter_ckpt = _resolve_checkpoint(
            args.inpainter_ckpt,
            project_root,
            "big-lama.pt",
        )
        if use_upscaler:
            _require_ckpt(upscaler_ckpt, "upscaler")
        if args.inpaint_mode in {"view", "uv"}:
            _require_ckpt(inpainter_ckpt, "inpainter")

        finisher = TexturePipeline(
            upscaler_ckpt_path=upscaler_ckpt if use_upscaler else None,
            inpaint_ckpt_path=inpainter_ckpt if args.inpaint_mode in {"view", "uv"} else None,
            device=args.device,
        )
        bake_config = ModProcessConfig(
            view_upscale=use_upscaler,
            inpaint_mode=args.inpaint_mode,
        )
        baked = finisher(
            mesh_path=str(mesh_path),
            save_dir=str(temp_dir),
            save_name="assembled",
            front_x=mesh_front_x,
            uv_unwarp=True,
            preprocess_mesh=args.preprocess_mesh,
            uv_size=spec.uv_size,
            rgb_path=str(packed_view_path),
            rgb_process_config=bake_config,
            camera_azimuth_deg=[0, 90, 180, 270, 180, 180],
        )

        shaded_path = Path(baked.shaded_model_save_path)
        shutil.copy2(shaded_path, output_path)

        if args.keep_intermediate:
            keep_dir = output_path.parent / f"{output_path.stem}_cache"
            if keep_dir.exists():
                shutil.rmtree(keep_dir)
            shutil.copytree(temp_dir, keep_dir)

        print(output_path)
    finally:
        temp_context.cleanup()


if __name__ == "__main__":
    main()
