#!/bin/bash
ssh -L 4444:localhost:8888 -L 4747:localhost:8787 -J faquista@gate.cloudveneto.it ubuntu@10.67.22.240