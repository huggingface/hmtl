#!/bin/bash
git pull

# Front
bower install
npm install
grunt
tsc

