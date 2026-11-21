#!/usr/bin/env bash
# Docker Engine inside WSL Ubuntu-22.04, official apt-repository method
# (https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive

# Remove conflicting unofficial packages if present
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
  apt-get remove -y "$pkg" 2>/dev/null || true
done

apt-get update
apt-get install -y ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Cross-compiling for the Kria (aarch64) may need arm64 emulation
apt-get install -y qemu-user-static binfmt-support

systemctl enable --now docker
docker version
docker run --rm hello-world
echo DOCKER_DONE
