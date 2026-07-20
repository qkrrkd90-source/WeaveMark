"""GPU check. Models load 4-bit (bitsandbytes), which is CUDA-only, so a missing
or arch-incompatible GPU should fail here rather than mid model-load."""
import torch

_NO_CUDA_MESSAGE = """\
WeaveMark requires a CUDA GPU.

torch.cuda.is_available() returned False. Common causes:

  1. CPU-only torch build. Check the version string:
         python -c "import torch; print(torch.__version__)"
     A CUDA build ends in '+cuXXX' (e.g. 2.7.1+cu128); a CPU-only wheel has no
     such suffix. `pip install torch` serves a CPU-only wheel on Windows, so
     install from the CUDA index instead -- see https://pytorch.org.

  2. No GPU visible to this process. Check `nvidia-smi`, and make sure
     CUDA_VISIBLE_DEVICES does not hide the device you intend to use.
"""

_ARCH_MISMATCH_MESSAGE = """\
This torch build has no CUDA kernels for your GPU.

  GPU:               {name} (compute capability sm_{cc})
  torch:             {version}
  kernels built for: {arch_list}

torch.cuda.is_available() returns True because the driver can see the card, but
any real kernel launch fails with 'no kernel image is available for execution on
the device'.

Install a torch build compiled for sm_{cc}. RTX 50-series (Blackwell, sm_120)
cards need torch >= 2.7 built against CUDA >= 12.8:

    pip uninstall -y torch
    pip install torch --index-url https://download.pytorch.org/whl/cu128

Pick the command matching your GPU and CUDA version at https://pytorch.org.
bitsandbytes must also ship kernels for your architecture, so upgrade it too:

    pip install -U bitsandbytes

Original error:
{error}
"""


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(_NO_CUDA_MESSAGE)

    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)

    # probe with a real kernel: is_available() is True even when the wheel has
    # no kernels for this arch (e.g. sm_120 on a cu121 build)
    try:
        probe = torch.zeros(8, device="cuda:0")
        probe.normal_()
        (probe * probe).sum().item()
        torch.cuda.synchronize()
    except RuntimeError as exc:
        raise RuntimeError(_ARCH_MISMATCH_MESSAGE.format(
            name=props.name,
            cc=f"{major}{minor}",
            version=torch.__version__,
            arch_list=" ".join(torch.cuda.get_arch_list()) or "(none)",
            error=exc,
        )) from exc

    print(f"[device] cuda:0 | {props.name} | sm_{major}{minor} | "
          f"{props.total_memory / 1024 ** 3:.1f} GiB")
    return torch.device("cuda:0")
