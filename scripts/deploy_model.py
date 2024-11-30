#!/usr/bin/env python3
"""Model deployment script."""

import argparse
import shutil
from pathlib import Path
from datetime import datetime

def deploy_model(model_path):
    """Deploy model to production directory."""
    try:
        # Deploy new model
        production_path = Path("models/production")
        production_path.mkdir(parents=True, exist_ok=True)

        # Archive current production model if exists
        current_model = production_path / "model.onnx"
        if current_model.exists():
            archive_path = production_path / "archive" / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.onnx"
            archive_path.parent.mkdir(exist_ok=True)
            shutil.move(str(current_model), str(archive_path))

        # Deploy new model
        shutil.copy(model_path, str(current_model))
        print(f"Model successfully deployed to production: {current_model}")
        return True

    except Exception as e:
        print(f"Deployment failed: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy model to production")
    parser.add_argument("--model-path", required=True, help="Path to model file")
    args = parser.parse_args()
    success = deploy_model(args.model_path)
    exit(0 if success else 1)
