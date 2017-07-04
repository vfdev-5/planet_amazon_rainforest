#!/usr/bin/env bash

echo "----------- Run all scripts -----------"

export ROOT=/output/planet_amazon_rainforest

bash 0_init_sources.bash

bash 1_start_training.bash