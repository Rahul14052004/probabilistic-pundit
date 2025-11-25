# CI/CD Explanation
- GitHub Actions will run tests and build an image
- The workflow authenticates to GCP using a service account JSON stored in GH Secrets
- The built image is pushed to Google Container Registry
- Deployment uses google-github-actions/deploy-cloudrun

Security notes: keep SA key in Secrets, use minimal IAM roles.
