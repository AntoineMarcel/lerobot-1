# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Teleop bimanuel : BiSOLeader (USB sur le PC) -> SolariaClient -> Pi (solaria_host + BiSOFollower).

Pi (adapter calibration_dir vers le dossier qui contient les JSON sur le Pi ; id = préfixe des fichiers) :
  python -m lerobot.robots.solaria.solaria_host \\
    --robot.type=bi_so_follower \\
    --robot.id=solaria_pi \\
    --robot.calibration_dir=/chemin/vers/calibration \\
    --robot.left_arm_config.port=/dev/ttyACM0 \\
    --robot.right_arm_config.port=/dev/ttyACM1

PC (depuis lerobot/, avec PYTHONPATH=src si besoin) :
  python examples/solaria/teleoperate.py
"""

import time
from pathlib import Path

from lerobot.robots.solaria import SolariaClient, SolariaClientConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30

# Répertoire des JSON `*_left.json` / `*_right.json` (parents[3] = racine du dépôt solaria).
_CALIB_DIR = Path(__file__).resolve().parents[3] / "calibration"


def main():
    robot_config = SolariaClientConfig(
        remote_ip="100.123.79.58",
        id="solaria",
    )
    teleop_config = BiSOLeaderConfig(
        id="solaria_leader",
        calibration_dir=_CALIB_DIR,
        left_arm_config=SOLeaderConfig(port="/dev/tty.usbmodem5B140296141"),
        right_arm_config=SOLeaderConfig(port="/dev/tty.usbmodem5B141132561"),
    )

    robot = SolariaClient(robot_config)
    leader = BiSOLeader(teleop_config)

    robot.connect()
    leader.connect()

    init_rerun(session_name="solaria_teleop")

    if not robot.is_connected or not leader.is_connected:
        raise RuntimeError("Robot ou téléop non connecté.")

    print("Boucle téléop (Ctrl+C pour arrêter)...")
    try:
        while True:
            t0 = time.perf_counter()
            observation = robot.get_observation()
            action = leader.get_action()
            _ = robot.send_action(action)
            log_rerun_data(observation=observation, action=action)
            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    finally:
        leader.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
