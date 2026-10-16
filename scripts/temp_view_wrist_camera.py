"""Show 20 augmented training images from the wrist camera on the robot's claw.

    python scripts/temp_view_wrist_camera.py
    python scripts/temp_view_wrist_camera.py --save .tmp/wrist_camera_samples.png --no-show

Uses the same shuffled, in-memory DataLoader and RandAugment as vision training.
All training shards are loaded up front. No PYTHONPATH setup is needed.
"""

from temp_view_vision_samples import main as view_samples


if __name__ == "__main__":
    view_samples(default_camera="wrist", description=__doc__)
