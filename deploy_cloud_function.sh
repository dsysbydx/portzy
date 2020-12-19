#!/bin/bash

gcloud functions deploy PortValueApi \
--entry-point handle \
--runtime python37 \
--trigger-http \
--allow-unauthenticated