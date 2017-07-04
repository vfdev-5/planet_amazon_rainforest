#!/usr/bin/env bash

echo "----------- Run all scripts -----------"

export ROOT=planet_amazon_rainforest
export DATA_PATH=/input

sh 0_init_sources.sh

sh 1_start_training.sh