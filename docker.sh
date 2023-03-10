#!/bin/sh

docker image build --tag kn-formation:latest .
docker container run \
        --name kn-formation \
        --rm \
        --interactive \
        --tty \
        --gpus all \
        --mount type=bind,src=$(pwd),dst=/codes \
        kn-formation:latest \
        bash
