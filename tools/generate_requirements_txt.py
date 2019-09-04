# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file outputs a requirements.txt based on the libraries defined in generate_conda_file.py
from generate_conda_file import (
    CONDA_BASE,
    CONDA_GPU,
    PIP_BASE,
    PIP_GPU,
    PIP_DARWIN,
    PIP_LINUX,
    PIP_WIN32,
    CONDA_DARWIN,
    CONDA_LINUX,
    CONDA_WIN32,
    PIP_DARWIN_GPU,
    PIP_LINUX_GPU,
    PIP_WIN32_GPU,
    CONDA_DARWIN_GPU,
    CONDA_LINUX_GPU,
    CONDA_WIN32_GPU,
)


if __name__ == "__main__":
    deps = list(CONDA_BASE.values())
    deps += list(CONDA_GPU.values())
    deps += list(PIP_BASE.values())
    deps += list(PIP_GPU.values())
    deps += list(PIP_DARWIN.values())
    deps += list(PIP_LINUX.values())
    deps += list(PIP_WIN32.values())
    deps += list(CONDA_DARWIN.values())
    deps += list(CONDA_LINUX.values())
    deps += list(CONDA_WIN32.values())
    deps += list(PIP_DARWIN_GPU.values())
    deps += list(PIP_LINUX_GPU.values())
    deps += list(PIP_WIN32_GPU.values())
    deps += list(CONDA_DARWIN_GPU.values())
    deps += list(CONDA_LINUX_GPU.values())
    deps += list(CONDA_WIN32_GPU.values())
    with open("requirements.txt", "w") as f:
        f.write("\n".join(set(deps)))

