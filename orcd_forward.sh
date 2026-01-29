#!/bin/bash
# Port forwarding script for PersonaPlex Apptainer container on MIT ORCD
# Usage: ./orcd_forward.sh <node number>
ssh -L 8998:node$1:8998 chang168@orcd-login.mit.edu
