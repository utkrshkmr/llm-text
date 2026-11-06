# Brief: Appendix E "Containerising ML Projects" (NEW)

Read first: BRIEF_conventions.md (§§1-2, 7-8), book/appendices/appD_setup.md (environment; link, do not repeat),
code/requirements.txt, code/setup.py, code/csrc/README.md, code/projects/*/README.md, .github/workflows/pages.yml.
Docker may not be installed on this box and you have no root: write and lint the files (hadolint if available via
pip/conda), and verify what can be verified without a daemon (e.g. `docker buildx` absent → explain; Apptainer can
build from a Dockerfile-derived def file without root if apptainer is installed; check). Do not install system packages.

Files you own: book/appendices/appE_containers.md {#app:containers}, figures appE_containers-<k>.svg,
code/docker/{Dockerfile (multi-stage: CUDA devel builder compiling llmsys._C for sm_80/sm_90 → slim runtime image
with the wheel; non-root user; pinned versions; build args for CUDA/torch versions), Dockerfile.dev (devcontainer),
.dockerignore, compose.yaml (serving the chat demo + engine with GPU reservation, healthcheck), apptainer.def,
devcontainer.json, Makefile targets, README.md}, code/tests/test_docker_files.py (static checks: pinned base image
digests or tags, no `latest`, non-root USER, .dockerignore excludes data/checkpoints, versions agree with
requirements.txt and torch CUDA major).

Content: why containers for ML (reproducibility, CUDA/driver split: what lives in the image vs on the host),
images and layers, the NVIDIA container toolkit (--gpus, CDI), choosing base images (CUDA runtime vs devel, PyTorch
NGC images), multi-stage builds for C++/CUDA extensions (TORCH_CUDA_ARCH_LIST, ccache, build cache mounts), Python
dependency pinning (lock files, uv/pip-tools), data and checkpoints as volumes, non-root and security basics, image
size hygiene, running multi-GPU and multi-node (torchrun in containers, NCCL env, shared memory size --shm-size/ipc),
compose for a serving stack, dev containers, Apptainer/Singularity on clusters without root, Kubernetes in brief
(device plugin, a Job manifest), CI building and testing images, and a checklist. Exercises and solutions.
