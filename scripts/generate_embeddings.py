"""Generate normalized dataset embeddings with the trained Temp Model.

S3.2 generates retrieval embeddings for a selected dataset split using the
best Sprint 2 checkpoint. Embeddings are written to the ignored embeddings/
directory; persistent storage format/metadata management is deferred to S3.3.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.temp_model.dataset import JewelrySplitDataset  # noqa: E402
from src.temp_model.model import EMBEDDING_DIM, INPUT_IMAGE_SIZE, CustomCNN  # noqa: E402

SPLITS = ("train", "validation", "test")

def resolve_device(requested: str) -> torch.device:
    if requested == "cpu": return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available(): raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_checkpoint(model: CustomCNN, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    if not checkpoint_path.is_file(): raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict): raise ValueError("Checkpoint must contain a model_state_dict mapping.")
    model.load_state_dict(state_dict)
    return checkpoint

def generate_split_embeddings(model, dataset, batch_size, num_workers, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    embeddings, metadata = [], []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Embedding", unit="batch"):
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            batch_embeddings = model.forward_retrieval(images)
            embeddings.append(batch_embeddings.cpu().numpy())
            for path, class_name, description in zip(batch["image_path"], batch["class"], batch["description"]):
                metadata.append({"image_path": str(path), "class": str(class_name), "description": str(description)})
    if not embeddings: raise RuntimeError("No embeddings were generated.")
    return np.concatenate(embeddings, axis=0).astype(np.float32, copy=False), metadata

def validate_embeddings(embeddings: np.ndarray, expected_count: int) -> None:
    if embeddings.shape != (expected_count, EMBEDDING_DIM): raise ValueError(f"Unexpected embedding shape: {embeddings.shape}; expected {(expected_count, EMBEDDING_DIM)}.")
    if not np.isfinite(embeddings).all(): raise ValueError("Generated embeddings contain NaN or Inf values.")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5, rtol=1e-5): raise ValueError("Generated embeddings are not L2-normalized.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Temp Model retrieval embeddings for a dataset split.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--manifest-dir", type=Path, default=REPO_ROOT / "results" / "dataset-split")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "results" / "training" / "best-checkpoint" / "best.pt")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "embeddings")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size <= 0: raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0: raise ValueError("--num-workers must be non-negative.")
    device = resolve_device(args.device)
    manifest_path = args.manifest_dir / f"{args.split}.csv"
    dataset = JewelrySplitDataset(manifest_path=manifest_path, dataset_root=args.dataset_root, image_size=INPUT_IMAGE_SIZE)
    model = CustomCNN()
    checkpoint = load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    embeddings, metadata = generate_split_embeddings(model, dataset, args.batch_size, args.num_workers, device)
    validate_embeddings(embeddings, len(dataset))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = args.output_dir / f"{args.split}.npy"
    metadata_path = args.output_dir / f"{args.split}_generation.json"
    np.save(embedding_path, embeddings)
    generation_metadata = {"generation_version": 1, "split": args.split, "sample_count": len(dataset), "embedding_dimension": EMBEDDING_DIM, "dtype": str(embeddings.dtype), "device": str(device), "checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint.get("epoch"), "normalized": True, "manifest": str(manifest_path), "metadata_records": metadata}
    with metadata_path.open("w", encoding="utf-8") as handle: json.dump(generation_metadata, handle, indent=2, ensure_ascii=False)
    print("S3.2 dataset embedding generation completed.")
    print(f"  Split: {args.split}")
    print(f"  Samples: {len(dataset)}")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Embedding dimension: {EMBEDDING_DIM}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"  Device: {device}")
    print(f"  Output: {embedding_path}")

if __name__ == "__main__": main()