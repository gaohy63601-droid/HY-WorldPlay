#!/usr/bin/env bash

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate worldplay
cd /workspace/WorldPolicy/HY-WorldPlay
export PYTHONPATH="/workspace/WorldPolicy/HY-WorldPlay:${PYTHONPATH:-}"
