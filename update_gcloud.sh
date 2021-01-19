#!/bin/sh
rsync -Pav -e "ssh -i $HOME/.ssh/google_compute_engine" . zhihua@34.90.228.238:/home/zhihua/framework/vison
