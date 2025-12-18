"""
Fix model configurations for sentence-transformers compatibility.
Adds missing modules.json, sentence_bert_config.json, and pooling config.

Usage:
    c:\other\defy\venv\Scripts\python.exe fix_model_configs.py
"""

import json
import os
from pathlib import Path

# Models that need fixing
MODELS_TO_FIX = [
    "models--emilyalsentzer--Bio_ClinicalBERT",
    "models--medicalai--ClinicalBERT",
    "models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "models--pritamdeka--S-PubMedBert-MS-MARCO",
]

# Template configs
MODULES_JSON = [
    {
        "idx": 0,
        "name": "0",
        "path": "",
        "type": "sentence_transformers.models.Transformer"
    },
    {
        "idx": 1,
        "name": "1",
        "path": "1_Pooling",
        "type": "sentence_transformers.models.Pooling"
    }
]

SENTENCE_BERT_CONFIG = {
    "max_seq_length": 512,
    "do_lower_case": False
}

CONFIG_SENTENCE_TRANSFORMERS = {
    "__version__": {
        "sentence_transformers": "2.2.2",
        "transformers": "4.34.0",
        "pytorch": "2.0.1"
    }
}

POOLING_CONFIG = {
    "word_embedding_dimension": 768,
    "pooling_mode_cls_token": False,
    "pooling_mode_mean_tokens": True,
    "pooling_mode_max_tokens": False,
    "pooling_mode_mean_sqrt_len_tokens": False
}


def fix_model(model_dir: Path):
    """Fix a single model's configuration."""
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        print(f"  No snapshots directory found")
        return False

    # Find snapshot directory
    snapshots = list(snapshots_dir.iterdir())
    if not snapshots:
        print(f"  No snapshots found")
        return False

    # Use first snapshot that has model files
    snapshot_dir = None
    for snap in snapshots:
        if snap.is_dir():
            files = list(snap.iterdir())
            if any(f.name.endswith('.safetensors') or f.name.endswith('.bin') for f in files):
                snapshot_dir = snap
                break

    if not snapshot_dir:
        print(f"  No valid snapshot found")
        return False

    print(f"  Fixing: {snapshot_dir.name}")

    # Check if config.json exists and has model_type
    config_path = snapshot_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        if "model_type" not in config:
            print(f"  WARNING: config.json missing model_type")

    # Create modules.json
    modules_path = snapshot_dir / "modules.json"
    if not modules_path.exists():
        with open(modules_path, 'w') as f:
            json.dump(MODULES_JSON, f, indent=2)
        print(f"  Created: modules.json")
    else:
        print(f"  Already exists: modules.json")

    # Create sentence_bert_config.json
    sbert_config_path = snapshot_dir / "sentence_bert_config.json"
    if not sbert_config_path.exists():
        with open(sbert_config_path, 'w') as f:
            json.dump(SENTENCE_BERT_CONFIG, f, indent=2)
        print(f"  Created: sentence_bert_config.json")
    else:
        print(f"  Already exists: sentence_bert_config.json")

    # Create config_sentence_transformers.json
    st_config_path = snapshot_dir / "config_sentence_transformers.json"
    if not st_config_path.exists():
        with open(st_config_path, 'w') as f:
            json.dump(CONFIG_SENTENCE_TRANSFORMERS, f, indent=2)
        print(f"  Created: config_sentence_transformers.json")
    else:
        print(f"  Already exists: config_sentence_transformers.json")

    # Create 1_Pooling directory and config
    pooling_dir = snapshot_dir / "1_Pooling"
    pooling_dir.mkdir(exist_ok=True)
    pooling_config_path = pooling_dir / "config.json"
    if not pooling_config_path.exists():
        with open(pooling_config_path, 'w') as f:
            json.dump(POOLING_CONFIG, f, indent=2)
        print(f"  Created: 1_Pooling/config.json")
    else:
        print(f"  Already exists: 1_Pooling/config.json")

    return True


def main():
    models_dir = Path(__file__).parent / "models"

    print("=" * 60)
    print("Fixing model configurations for sentence-transformers")
    print("=" * 60)

    fixed = 0
    for model_name in MODELS_TO_FIX:
        model_path = models_dir / model_name
        print(f"\nProcessing: {model_name}")

        if not model_path.exists():
            print(f"  Model not found, skipping")
            continue

        if fix_model(model_path):
            fixed += 1

    print(f"\n{'=' * 60}")
    print(f"Fixed {fixed}/{len(MODELS_TO_FIX)} models")
    print("=" * 60)


if __name__ == "__main__":
    main()
