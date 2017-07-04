#!/bin/sh

echo "----------- Run all scripts -----------"

export ROOT=/output/planet_amazon_rainforest

sh 0_init_sources.sh

sh 1_start_training.sh