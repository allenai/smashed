#!/usr/bin/env bash

autopep8 --in-place --recursive . && isort . && black .
