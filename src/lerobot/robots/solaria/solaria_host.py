#!/usr/bin/env python

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

"""Run on the Raspberry Pi: exposes a BiSOFollower over ZMQ (commands in, observations out)."""

import base64
import json
import logging
import time

import cv2
import draccus
import numpy as np
import zmq

from ..bi_so_follower import BiSOFollower
from ..utils import make_robot_from_config
from .config_solaria import SolariaHostConfig, SolariaServerConfig


class SolariaHost:
    def __init__(self, config: SolariaHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


def _observation_to_json_serializable(obs: dict) -> dict:
    out = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim >= 2:
            ret, buffer = cv2.imencode(".jpg", value, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ret:
                out[key] = base64.b64encode(buffer).decode("utf-8")
            else:
                out[key] = ""
        else:
            out[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value
    return out


@draccus.wrap()
def main(cfg: SolariaServerConfig):
    logging.basicConfig(level=logging.INFO)
    logging.info("Configuring Solaria (BiSOFollower host)")
    robot = make_robot_from_config(cfg.robot)
    if not isinstance(robot, BiSOFollower):
        raise ValueError(
            f"solaria_host attend --robot.type=bi_so_follower, reçu type={cfg.robot.type!r}."
        )

    logging.info("Connecting follower arms on Pi")
    robot.connect()

    logging.info("Starting ZMQ host (cmd PULL %s, obs PUSH %s)", cfg.host.port_zmq_cmd, cfg.host.port_zmq_observations)
    host = SolariaHost(cfg.host)

    last_cmd_time = time.time()
    watchdog_fired = False
    logging.info("Waiting for commands...")
    try:
        start = time.perf_counter()
        duration = 0.0
        while duration < host.connection_time_s:
            loop_start_time = time.time()
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                robot.send_action(data)
                last_cmd_time = time.time()
                watchdog_fired = False
            except zmq.Again:
                pass
            except Exception as e:
                logging.error("Command handling failed: %s", e)

            now = time.time()
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000.0) and not watchdog_fired:
                logging.warning(
                    "No command for more than %s ms; arms hold last goal (no base to stop on Solaria).",
                    host.watchdog_timeout_ms,
                )
                watchdog_fired = True

            last_observation = robot.get_observation()
            payload = _observation_to_json_serializable(last_observation)

            try:
                host.zmq_observation_socket.send_string(json.dumps(payload), flags=zmq.NOBLOCK)
            except zmq.Again:
                logging.debug("Dropping observation, no client connected")

            elapsed = time.time() - loop_start_time
            time.sleep(max(1.0 / host.max_loop_freq_hz - elapsed, 0.0))
            duration = time.perf_counter() - start

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt, exiting...")
    finally:
        logging.info("Shutting down Solaria host")
        robot.disconnect()
        host.disconnect()

    logging.info("Solaria host finished")


if __name__ == "__main__":
    main()
