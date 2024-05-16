#!/bin/bash
set -e

# add ssh key
eval `ssh-agent -s` && ssh-add ~/.ssh/ssh_key1 && ssh-add ~/.ssh/ssh_key2
# eval `ssh-agent -s` && ssh-add ~/.ssh/ssh_key1 