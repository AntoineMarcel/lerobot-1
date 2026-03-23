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

"""Enregistrement téléop : BiSOLeader (PC) + SolariaClient -> dataset.

Prérequis : `solaria_host` actif sur le Pi (voir teleoperate.py).
"""

import logging
from pathlib import Path

from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_processors
from lerobot.robots.solaria import SolariaClient, SolariaClientConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

NUM_EPISODES = 20
FPS = 30
EPISODE_TIME_SEC = 30
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "Put the sock in the box"
HF_REPO_ID = "SoSolaris/put_the_sock_in_the_box"
_CALIB_DIR = Path(__file__).resolve().parents[3] / "calibration"

def main():
    logging.basicConfig(level=logging.INFO)

    robot_config = SolariaClientConfig(
        remote_ip="100.123.79.58",
        id="solaria",
    )
    leader_config = BiSOLeaderConfig(
        id="solaria_leader",
        calibration_dir=_CALIB_DIR,
        left_arm_config=SOLeaderConfig(port="/dev/tty.usbmodem5B140296141"),
        right_arm_config=SOLeaderConfig(port="/dev/tty.usbmodem5B141132561"),
    )

    robot = SolariaClient(robot_config)
    leader = BiSOLeader(leader_config)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

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

    listener, events = init_keyboard_listener()
    init_rerun(session_name="solaria_record")

    try:
        if not robot.is_connected or not leader.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        print("Starting record loop...")
        recorded_episodes = 0
        while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"Recording episode {recorded_episodes}")

            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                dataset=dataset,
                teleop=leader,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

            if not events["stop_recording"] and (
                (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop=leader,
                    control_time_s=RESET_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    display_data=True,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            recorded_episodes += 1
    finally:
        log_say("Stop recording")
        robot.disconnect()
        leader.disconnect()
        listener.stop()

        dataset.finalize()
        dataset.push_to_hub()


if __name__ == "__main__":
    main()
