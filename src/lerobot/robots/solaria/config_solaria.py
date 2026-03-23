# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class SolariaHostConfig:
    port_zmq_cmd: int = 5557
    port_zmq_observations: int = 5558

    connection_time_s: int = 86_400

    watchdog_timeout_ms: int = 500

    max_loop_freq_hz: int = 30


@RobotConfig.register_subclass("solaria_client")
@dataclass
class SolariaClientConfig(RobotConfig):
    """Client-side robot: ZMQ bridge to a BiSOFollower running on a Raspberry Pi (solaria host)."""

    remote_ip: str

    port_zmq_cmd: int = 5557
    port_zmq_observations: int = 5558

    left_cameras: dict[str, CameraConfig] = field(default_factory=dict)
    right_cameras: dict[str, CameraConfig] = field(default_factory=dict)

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5


@dataclass
class SolariaServerConfig:
    """CLI config for `python -m lerobot.robots.solaria.solaria_host`."""

    # RobotConfig (choice) so `--robot.type=bi_so_follower` is accepted by draccus.
    robot: RobotConfig
    host: SolariaHostConfig = field(default_factory=SolariaHostConfig)
