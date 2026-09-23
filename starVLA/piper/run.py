"""Synchronous Piper/Piper-X deployment; observation-only unless --execute is given."""

import argparse
import json
import signal
import threading
import time
from contextlib import ExitStack
from pathlib import Path

from starVLA.piper.adapter import MockBackend, MockPolicy, PiperAdapter, SyncRunner
from starVLA.piper.client import PiperWebsocketPolicy
from starVLA.piper.native_x import PiperXAdapter, PiperXBackend
from starVLA.piper.ros import RosBackend, RosTopics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rung", choices=("s0", "s2", "s2_pair"), required=True)
    p.add_argument("--robot", choices=("piper", "piper_x"), default="piper")
    p.add_argument("--backend", choices=("mock", "ros", "sdk"), default="mock")
    p.add_argument("--policy", choices=("mock", "websocket"), default="mock")
    p.add_argument("--prompt", required=True)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--execute-steps", type=int, default=10)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--gripper-max", type=float)
    p.add_argument("--ros-topics", type=Path)
    p.add_argument("--left-can", default="can0")
    p.add_argument("--right-can", default="can1")
    p.add_argument("--firmware", choices=("default", "v183", "v188", "v189"), default="default")
    p.add_argument("--camera-serials", type=Path)
    p.add_argument("--speed-percent", type=int, default=30)
    p.add_argument("--gripper-force", type=float, default=1.0)
    p.add_argument("--max-observation-age", type=float, default=0.5)
    p.add_argument("--max-observation-skew", type=float, default=0.04)
    p.add_argument("--observation-timeout", type=float, default=5)
    p.add_argument("--log", type=Path)
    args = p.parse_args()
    if args.cycles < 0 or not 1 <= args.execute_steps <= 50:
        p.error("cycles must be nonnegative; execute-steps must be 1..50")
    if args.execute and (args.backend == "mock" or args.policy != "websocket"):
        p.error("--execute requires a hardware backend and a WebSocket policy")
    if args.backend == "sdk" and (args.robot != "piper_x" or args.camera_serials is None):
        p.error("SDK backend requires --robot piper_x and --camera-serials")
    if args.robot == "piper_x" and args.backend != "sdk":
        p.error("Native Piper-X needs recorded SDK flange feedback; select --backend sdk")
    if args.backend == "ros" and args.robot != "piper":
        p.error("ROS backend uses Piper FK")
    if args.backend != "mock" and args.policy != "websocket":
        p.error("Hardware observations require a matching WebSocket policy")
    adapter_class = PiperXAdapter if args.robot == "piper_x" else PiperAdapter
    gripper_max = args.gripper_max if args.gripper_max is not None else (0.10 if args.robot == "piper_x" else 0.08)
    adapter = adapter_class(args.rung, prompt=args.prompt, gripper_max=gripper_max)
    stop = threading.Event()

    def halt(_signal, _frame):
        stop.set()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, halt)
    signal.signal(signal.SIGTERM, halt)
    log = args.log or Path("results/Piper/deployment") / f"{args.robot}_{args.rung}_{time.time_ns()}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        output = stack.enter_context(log.open("x"))
        policy = (
            MockPolicy(args.rung)
            if args.policy == "mock"
            else PiperWebsocketPolicy(args.host, args.port, timeout=args.timeout)
        )
        stack.callback(policy.close)
        metadata = policy.get_server_metadata()
        adapter.validate_metadata(metadata)
        if metadata.get("rtc_mode") != "none":
            raise ValueError("This client requires synchronous checkpoints")
        if args.execute and metadata.get("smoke_only", False):
            raise ValueError("Smoke checkpoints are for pipeline validation, not hardware execution")
        common = dict(
            max_age=args.max_observation_age,
            max_skew=args.max_observation_skew,
            observation_timeout=args.observation_timeout,
            execute=args.execute,
        )
        if args.backend == "mock":
            backend = MockBackend()
        elif args.backend == "ros":
            topics = RosTopics(**json.loads(args.ros_topics.read_text())) if args.ros_topics else RosTopics()
            backend = RosBackend(topics=topics, **common)
        else:
            backend = PiperXBackend(
                left_can=args.left_can,
                right_can=args.right_can,
                camera_serials=json.loads(args.camera_serials.read_text()),
                firmware=args.firmware,
                speed_percent=args.speed_percent,
                gripper_force=args.gripper_force,
                gripper_max=gripper_max,
                **common,
            )
        stack.callback(backend.close)
        runner = SyncRunner(
            adapter,
            backend,
            backend,
            policy,
            execute_steps=args.execute_steps,
            max_observation_age=args.max_observation_age,
            should_stop=stop.is_set,
        )
        if args.backend == "sdk":
            backend.start_execution()
        try:
            cycle = 0
            while not stop.is_set() and (args.cycles == 0 or cycle < args.cycles):
                record = runner.cycle()
                record.update(cycle=cycle, execute=args.execute, robot=args.robot, server_metadata=metadata)
                output.write(json.dumps(record, allow_nan=False) + "\n")
                output.flush()
                print(f"Cycle {cycle}: {record['mode']}, {record['executed_steps']} targets, execute={args.execute}")
                cycle += 1
        except (KeyboardInterrupt, InterruptedError):
            print("Stopped command publication.")


if __name__ == "__main__":
    main()
