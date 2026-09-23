# Deploying VLAct Piper policies

The GPU server runs the trained QwenPI model. The CPU robot client sends three
raw RGB views, an unnormalized state and the training task prompt, receives
unnormalized encoded actions, and decodes them into absolute commands.
Training and serving share coordinate and normalization code.

This is synchronous chunk execution: hold while inference runs, then publish
the first 10 of 50 predicted targets at 30 Hz, capture a new observation, and
repeat. All 50 targets use the same observation anchor, predicting t+1..t+50.
Inference time is additional to the publication interval. There is no RTC,
prefix skipping, action interpolation or uninterrupted-30-Hz claim.

## GPU policy server

From the VLAct checkout root:

```bash
.venv/bin/python -m starVLA.piper.serve \
  --checkpoint results/Piper/vlact_piper_s2_pair_v1/step_060000 \
  --host 0.0.0.0 --port 8000
```

Select the matching native or aligned run for other experiments. A complete
checkpoint directory contains `model.safetensors`, `complete.json`,
`config.yaml`, `contract.json`, and `norm_stats.json`; full-training checkpoints
also contain optimizer/scheduler/RNG state. Keep the local Qwen3-VL base assets
and Qwen3-0.6B directory available on the GPU host because model construction
uses their architecture/tokenizer. Relocate them with `--vlm` and `--expert`.
The trained checkpoint strictly replaces model weights. Serving needs no
source videos or training cache.

The handshake advertises rung, 14/23 state dimensions, 14/20 action dimensions,
frame, robot, horizon, t+1 offset, three camera keys, nominal base spacing,
normalization fingerprint and smoke status. The client checks representation
and robot/frame before constructing a hardware backend. The server validates
state shape, finite values and RGB type, and uses checkpoint normalization.
The server accepts variable-resolution RGB and shares training's 224x224 resize.
The 640x480 handshake field describes the robot clients' acquisition contract;
it is not an assertion about source videos. Inspected Piper-X videos are
1280x720 despite stale 640x480 metadata.
Do not apply normalization again in a client or send encoded S2 actions to
an absolute-joint executor.

The socket is a plain MessagePack/WebSocket service. Bind it to an interface
reachable by the robot on the deployment network; no authentication layer is
provided by this command.

## Portable CPU robot client

Export on the training server:

```bash
.venv/bin/python examples/Piper/export_client.py \
  --output results/Piper/client_bundle
```

Copy `results/Piper/client_bundle.tar.gz` to the robot host, extract and enter
the directory. This bundle contains only NumPy/SciPy geometry, ROS/SDK adapters,
WebSocket/MessagePack transport, configuration and source licenses. It does
not require Torch, Transformers, kai0, or source datasets.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-client.txt
```

For Piper ROS deployment, use an interpreter/environment that can also import
`rospy`, `sensor_msgs.msg` and the robot workspace's `piper_msgs.msg`, and source
that workspace. A separate virtual environment must be given access to those
ROS packages. Python 3.10+ is required by this client.

The examples below use paths relative to the exported bundle. Inside the
VLAct checkout, prefix the JSON paths with `examples/Piper/`.

## Piper with ROS

First read actual observations and log predictions:

```bash
python -m starVLA.piper.run --robot piper --rung s2_pair \
  --backend ros --policy websocket --host GPU_HOST --port 8000 \
  --prompt 'Task: fold the cloth. Scene: internal. Type: teleop. Quality: 5.' \
  --ros-topics ros_topics.json --cycles 1 --execute-steps 10
```

Use `--rung s0` or `s2` with the corresponding server. Aligned Piper-X or mixed
Piper/aligned-X models also use `--robot piper`; set the recorded prompt for
the task, such as `Stack the three bowls together.` No client-side frame
transfer is needed for aligned checkpoints.

| Stream | Default topic |
| --- | --- |
| Head RGB | `/camera_f/color/image_raw` |
| Left wrist RGB | `/camera_l/color/image_raw` |
| Right wrist RGB | `/camera_r/color/image_raw` |
| Left/right follower joints | `/puppet/joint_left`, `/puppet/joint_right` |
| S0 command | `/master/joint_left`, `/master/joint_right` |
| S2/S2-pair command | `/pos_cmd_left`, `/pos_cmd_right` |

RGB input must be 640x480 (ROS `rgb8` or `bgr8`, decoded to RGB). Joint feedback
contains six joint angles in radians plus full opening in metres per arm.
When present, names must be `joint0`..`joint6` and are reordered. Empty names
mean values are already in that order. Header timestamps must share a clock
and be positive. Defaults allow 40 ms cross-stream skew and 0.5 s observation
age, with a 5 s acquisition timeout; override the corresponding observation
flags when the actual sensor system requires it.

To publish motion on a prepared robot, add `--execute`; `--cycles 0` continues
until interrupted. The observation-only mode creates no motion publishers.
Execution requires command-topic subscribers. S0 emits absolute `JointState`;
EEF rungs emit absolute own-base flange xyz and XYZ Euler angles via `PosCmd`.
The ROS driver owns IK and conversion from SI to CAN units. No robot was used
to validate driver mapping, IK or tracking here. No enable/home command is
sent by the ROS backend.

## Native Piper-X with SDK and RealSense

In the bundle, install hardware dependencies and set the three camera serials:

```bash
python -m pip install -r requirements-piper-x.txt
cp camera_serials.example.json camera_serials.json
```

The requirements pin the same pyAgxArm revision used by the kai0 reference.
Configure the cell's two SocketCAN interfaces using its existing bring-up
procedure. The client selects `ArmModel.PIPER_X`, reads native flange feedback,
and never applies Piper FK. Firmware choices are explicit in `--help`; match
the installed controller rather than changing the default blindly.

```bash
python -m starVLA.piper.run --robot piper_x --rung s2_pair \
  --backend sdk --policy websocket --host GPU_HOST --port 8000 \
  --prompt 'Stack the three bowls together.' \
  --left-can can0 --right-can can1 --firmware default \
  --camera-serials camera_serials.json --cycles 1
```

S0 and S2 also work with their native-X servers. A native-X policy must not be
served to the Piper client, and an aligned policy must not be used with this
native-X client. Without `--execute`, the SDK backend only reads/logs.
With `--execute`, it enables both arms after fresh feedback is available,
sets the requested speed (default 30%), and sends `move_j` for S0 or `move_p`
for EEF, plus opening commands (default force 1 N). It never homes the arms.

## Execution boundaries and validation

Grippers are clipped to `[0, gripper_max]`, default 0.08 m for Piper and
0.10 m for Piper-X, and logs record clipping counts. Nonfinite output,
wrong shapes, degenerate rotation rows, old-episode responses and mismatched
metadata are rejected. There is no collision planner or general workspace/
joint-limit checker. Stopping ends further publication and closes transports;
it does not cancel targets already accepted by the robot controller.

Smoke checkpoints are deliberately refused with `--execute`. They may be
served and inspected offline. Use the mock transport without hardware:

```bash
python -m starVLA.piper.run --rung s2_pair --prompt 'Fold the cloth.' \
  --cycles 1 --execute-steps 2
```

To validate a saved model on real held-out images, from the training checkout:

```bash
PYTHONPATH="$PWD" .venv/bin/python examples/Piper/check_checkpoint.py \
  --checkpoint results/Piper/registered_pretrained_pair_smoke_20260922/step_000002 \
  --cache playground/Piper/registered_smoke/mixtures/piper_all_aligned --rung s2_pair \
  --output results/Piper/validation/pair_server.json
```

This starts a temporary localhost server, checks metadata, sends real held-out
images over WebSocket, verifies finite actions, computes physical-space errors
and checks rejection of malformed state. It never creates ROS/CAN clients.
See the validation document in the checkout for actual results and limits.
