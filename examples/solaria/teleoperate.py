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

import logging
import time
from pathlib import Path

from lerobot.robots.solaria import SolariaClient, SolariaClientConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30

_CALIB_DIR = Path("/Users/antoinemarcel/Desktop/test/solaria/calibration/")


def main():
    logging.basicConfig(level=logging.INFO)

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
    leader.connect(calibrate=False)
    for arm in (leader.left_arm, leader.right_arm):
        if not arm.is_calibrated:
            if arm.calibration:
                logging.info("Writing calibration from file for leader arm id=%s", arm.id)
                with arm.bus.torque_disabled():
                    arm.bus.write_calibration(arm.calibration)
                if not arm.is_calibrated:
                    logging.warning("Motors may still not match calibration file for id=%s", arm.id)
            else:
                raise RuntimeError(
                    f"Missing calibration JSON for {arm.id}: {arm.calibration_fpath}. "
                    "Create the file or run a one-off `leader.connect()` with calibration on a TTY."
                )

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
