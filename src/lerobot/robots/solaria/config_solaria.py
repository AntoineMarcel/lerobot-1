# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from ..config import RobotConfig


_CAMERA_ENV = (
    ("left", "CAMERA_LEFT_PORT", "/dev/video2"),
    ("right", "CAMERA_RIGHT_PORT", "/dev/video0"),
    ("up", "CAMERA_UP_PORT", "/dev/video4"),
)


def _index_or_path(raw: str) -> int | Path:
    return int(raw) if raw.isdigit() else Path(raw)


def solaria_cameras_config() -> dict[str, CameraConfig]:
    """Mêmes clés/résolutions sur Pi (OpenCV) et PC (shapes seulement, le path n'est pas ouvert)."""
    fps = int(os.getenv("CAMERA_FPS") or "10")
    names = {n.strip() for n in (os.getenv("CAMERA_NAMES") or "up,right").split(",") if n.strip()}
    cameras: dict[str, CameraConfig] = {}
    for name, env_key, default in _CAMERA_ENV:
        if name not in names:
            continue
        cameras[name] = OpenCVCameraConfig(
            index_or_path=_index_or_path(os.getenv(env_key) or default),
            fps=fps,
            width=640,
            height=480,
            fourcc="MJPG",
            warmup_s=0,
        )
    return cameras


@dataclass
class SolariaHostConfig:
    port_zmq_cmd: int = 5557
    port_zmq_observations: int = 5558

    connection_time_s: int = 86_400

    watchdog_timeout_ms: int = 500

    max_loop_freq_hz: int = 30

    cameras: dict[str, CameraConfig] = field(default_factory=solaria_cameras_config)


@RobotConfig.register_subclass("solaria_client")
@dataclass
class SolariaClientConfig(RobotConfig):
    """Client-side robot: ZMQ bridge to SO follower arms running on a Raspberry Pi."""

    remote_ip: str

    port_zmq_cmd: int = 5557
    port_zmq_observations: int = 5558

    # "single" for so100_follower/so101_follower, "bimanual" for bi_so_follower.
    arm_mode: str = "bimanual"

    # Same keys and resolution as the host; OpenCV paths are not used on the PC.
    cameras: dict[str, CameraConfig] = field(default_factory=solaria_cameras_config)

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5


@dataclass
class SolariaServerConfig:
    """CLI config for `python -m lerobot.robots.solaria.solaria_host`."""

    # RobotConfig (choice) accepts `so101_follower`, `so100_follower`, or `bi_so_follower`.
    robot: RobotConfig
    host: SolariaHostConfig = field(default_factory=SolariaHostConfig)
