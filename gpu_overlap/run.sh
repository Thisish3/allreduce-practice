#!/bin/bash
# Run overlap.py 3 times on the GPU inside the vLLM image (for torch), via k3s containerd.
# Then: python summarize.py
REPO=$(cd "$(dirname "$0")" && pwd)
IMG=docker.io/vllm/vllm-openai:v0.11.2
K="k3s kubectl"
systemctl start k3s
for i in $(seq 1 60); do $K get nodes 2>/dev/null | grep -q ' Ready' && break; sleep 2; done
$K scale deployment vllm-3080 --replicas=0 >/dev/null 2>&1
for i in $(seq 1 60); do [ -z "$($K get pods -l app=vllm-3080 --no-headers 2>/dev/null)" ] && break; sleep 2; done
echo "[$(date +%H:%M:%S)] GPU free"
for r in 1 2 3; do
  k3s ctr -n k8s.io run --rm --net-host --cwd /work --runc-binary /usr/bin/nvidia-container-runtime --env NVIDIA_VISIBLE_DEVICES=all --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --mount type=bind,src=$REPO,dst=/work,options=rbind:rw --mount type=bind,src=/dev/shm,dst=/dev/shm,options=rbind:rw $IMG overlap$r python3 overlap.py $r 2>&1 | grep -v -i warn
  echo "[$(date +%H:%M:%S)] run $r exit=${PIPESTATUS[0]}"
done
systemctl stop k3s
