#!/usr/bin/env python
"""Simple Feetech motor ID tool.

R + ENTER  -> read the ID (and baudrate) of the connected motor
S + ENTER  -> set a new ID (then type the number + ENTER)
Z + ENTER  -> rotate the motor 2s one way, then 2s the other way
Q + ENTER  -> quit
"""

import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.feetech.feetech import OperatingMode

PORT = "/dev/tty.usbmodem5B141122741"
MODEL = "sts3215"
SPIN_SPEED = 1500


def find_motor():
    """Return (baudrate, id) of the connected motor, or (None, None)."""
    bus = FeetechMotorsBus(PORT, {})
    bus._connect(handshake=False)
    try:
        for baudrate in bus.available_baudrates:
            bus.set_baudrate(baudrate)
            found = bus.broadcast_ping()
            if found:
                id_ = next(iter(found))
                return baudrate, id_
        return None, None
    finally:
        bus.port_handler.closePort()


def read_motor():
    bus = FeetechMotorsBus(PORT, {})
    bus._connect(handshake=False)
    try:
        for baudrate in bus.available_baudrates:
            bus.set_baudrate(baudrate)
            found = bus.broadcast_ping()
            if found:
                for id_, model_nb in found.items():
                    print(f"  -> ID={id_}  model_number={model_nb}  baudrate={baudrate}")
                return
        print("  No motor found. Check the connection.")
    finally:
        bus.port_handler.closePort()


def rotate_motor():
    baudrate, id_ = find_motor()
    if id_ is None:
        print("  No motor found. Check the connection.")
        return

    bus = FeetechMotorsBus(PORT, {"motor": Motor(id_, MODEL, MotorNormMode.RANGE_M100_100)})
    bus._connect(handshake=False)
    try:
        bus.set_baudrate(baudrate)
        bus.disable_torque()
        bus.write("Operating_Mode", "motor", OperatingMode.VELOCITY.value, normalize=False)
        bus.enable_torque()
        print(f"  -> Motor ID={id_} spinning forward 2s...")
        bus.write("Goal_Velocity", "motor", SPIN_SPEED, normalize=False)
        time.sleep(2)
        print("  -> spinning backward 2s...")
        bus.write("Goal_Velocity", "motor", -SPIN_SPEED, normalize=False)
        time.sleep(2)
        bus.write("Goal_Velocity", "motor", 0, normalize=False)
        bus.disable_torque()
        bus.write("Operating_Mode", "motor", OperatingMode.POSITION.value, normalize=False)
    finally:
        if bus.is_connected:
            bus.port_handler.closePort()


def setup_motor(target_id):
    bus = FeetechMotorsBus(PORT, {"motor": Motor(target_id, MODEL, MotorNormMode.RANGE_M100_100)})
    try:
        bus.setup_motor("motor")
        print(f"  -> Motor ID set to {target_id} (baudrate {bus.default_baudrate}).")
    finally:
        if bus.is_connected:
            bus.port_handler.closePort()


def main():
    print(__doc__)
    while True:
        choice = input("R (read) / S (setup) / Z (rotate) / Q (quit) > ").strip().lower()
        if choice == "r":
            read_motor()
        elif choice == "s":
            raw = input("  New ID > ").strip()
            if not raw.isdigit():
                print("  Invalid ID.")
                continue
            setup_motor(int(raw))
        elif choice == "z":
            rotate_motor()
        elif choice == "q":
            break
        else:
            print("  Type R, S, Z or Q.")


if __name__ == "__main__":
    main()
