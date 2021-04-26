#!/bin/sh
rsync -Pav -e "ssh -i $HOME/.ssh/google_compute_engine" . zhihua@10.164.0.99:/home/zhihua/work/deepspace
