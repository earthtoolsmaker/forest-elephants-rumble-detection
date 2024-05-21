#!/usr/bin/env bash

set -x

rclone --drive-shared-with-me --progress copy remote:"AI for Forest Elephants 2 Shared/01. Data" "data/01_raw/"
