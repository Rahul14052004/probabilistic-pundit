#!/usr/bin/env bash
set -e
PROJECT=$1
REGION=${2:-us-central1}
IMAGE=gcr.io/${PROJECT}/prob-pundit:latest

gcloud builds submit --tag $IMAGE

gcloud run deploy prob-pundit --image $IMAGE --region $REGION --platform managed --allow-unauthenticated
