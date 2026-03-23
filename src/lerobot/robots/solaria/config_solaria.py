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
from dotenv import load_dotenv
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from ..config import RobotConfig

load_dotenv()


def solaria_cameras_config() -> dict[str, CameraConfig]:
    """Three views on the Pi: left, right, up (ZMQ observation keys).

    Set ``CAMERA_LEFT_PORT``, ``CAMERA_RIGHT_PORT``, ``CAMERA_UP_PORT`` to override devices.
    Defaults: ``/dev/video0``, ``/dev/video2``, ``/dev/video4`` (typical multi-UVC on Linux).
    """
    return {
        "left": OpenCVCameraConfig(
            index_or_path=os.getenv("CAMERA_LEFT_PORT"),
            fps=30,
            width=640,
            height=480,
            fourcc="MJPG",
        ),
        "right": OpenCVCameraConfig(
            index_or_path=os.getenv("CAMERA_RIGHT_PORT"),
            fps=30,
            width=640,
            height=480,
            fourcc="MJPG",
        ),
        "up": OpenCVCameraConfig(
            index_or_path=os.getenv("CAMERA_UP_PORT"),
            fps=30,
            width=640,
            height=480,
            fourcc="MJPG",
        ),
    }


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
    """Client-side robot: ZMQ bridge to a BiSOFollower running on a Raspberry Pi (solaria host)."""

    remote_ip: str

    port_zmq_cmd: int = 5557
    port_zmq_observations: int = 5558

    # Same keys and resolution as the host; OpenCV paths are not used on the PC.
    cameras: dict[str, CameraConfig] = field(default_factory=solaria_cameras_config)

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5


@dataclass
class SolariaServerConfig:
    """CLI config for `python -m lerobot.robots.solaria.solaria_host`."""

    # RobotConfig (choice) so `--robot.type=bi_so_follower` is accepted by draccus.
    robot: RobotConfig
    host: SolariaHostConfig = field(default_factory=SolariaHostConfig)
