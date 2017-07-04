#!/bin/sh

DATA_ID=""

echo "- RUN FLOYD -"
floyd run --gpu --data DATA_ID "sh start_all.sh"