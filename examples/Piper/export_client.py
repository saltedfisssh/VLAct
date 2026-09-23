"""Export the CPU-only Piper client without the training package dependencies."""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    destination = args.output.resolve()
    if destination.exists() or destination.with_suffix(".tar.gz").exists():
        raise FileExistsError(destination)
    modules = (
        "__init__.py",
        "geometry.py",
        "transfer.py",
        "adapter.py",
        "ros.py",
        "native_x.py",
        "client.py",
        "msgpack_numpy.py",
        "run.py",
        "NOTICE.md",
        "LICENSE.kai0",
    )
    package = destination / "starVLA/piper"
    package.mkdir(parents=True)
    (destination / "starVLA/__init__.py").write_text("")
    for name in modules:
        shutil.copyfile(root / "starVLA/piper" / name, package / name)
    for name in (
        "requirements-client.txt",
        "requirements-piper-x.txt",
        "ros_topics.json",
        "camera_serials.example.json",
    ):
        shutil.copyfile(root / "examples/Piper" / name, destination / name)
    shutil.copyfile(root / "docs/piper_deployment.md", destination / "README.md")
    print(shutil.make_archive(str(destination), "gztar", root_dir=destination.parent, base_dir=destination.name))


if __name__ == "__main__":
    main()
