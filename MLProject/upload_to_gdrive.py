name: CI Workflow

on:
  push:
    branches: ["main"]
  workflow_dispatch:

permissions:
  contents: write

jobs:
  build:
    runs-on: ubuntu-latest

    env:
      MLFLOW_TRACKING_URI: file:${{ github.workspace }}/mlruns
      MLFLOW_EXPERIMENT_NAME: CI-Workflow

    steps:
      # 1. Checkout repository
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          lfs: true

      # 2. Setup Git LFS
      - name: Setup Git LFS
        run: |
          git lfs install

      # 3. Setup Python
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12.7"

      # 4. Install dependencies
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install \
            "mlflow==2.19.0" \
            "catboost==1.2.5" \
            "scikit-learn==1.5.2" \
            "pandas==2.2.3" \
            "numpy==1.26.4" \
            "matplotlib==3.9.2" \
            "scipy==1.11.4"

      # 5. Jalankan MLflow project (output → mlruns/ di root repo)
      - name: Run MLflow Project
        env:
          MLFLOW_TRACKING_URI: ${{ env.MLFLOW_TRACKING_URI }}
        working-directory: MLProject
        run: |
          echo "MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"
          mlflow run . --env-manager=local --experiment-name "${MLFLOW_EXPERIMENT_NAME}"

      # 6. Ambil RUN_ID dengan model terbaru
      - name: Get latest run ID
        run: |
          MODEL_PATH=$(find mlruns -type f -path "*/artifacts/model/MLmodel" -printf '%T@ %p\n' \
                        | sort -nr | head -n1 | awk '{print $2}')

          if [ -z "$MODEL_PATH" ]; then
            echo "RUN_ID=" >> $GITHUB_ENV
            echo "DOCKER_BUILT=false" >> $GITHUB_ENV
            exit 0
          fi

          RUN_DIR=$(dirname "$(dirname "$(dirname "$MODEL_PATH")")")
          RUN_ID=$(basename "$RUN_DIR")

          echo "RUN_ID=$RUN_ID" >> $GITHUB_ENV
          echo "DOCKER_BUILT=false" >> $GITHUB_ENV

      # 7. Commit MLflow artifacts using Git LFS
      - name: Commit MLflow Artifacts with Git LFS
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"

          git lfs track "mlruns/**"
          git add .gitattributes
          git add mlruns || true

          git commit -m "Add MLflow artifacts [skip ci]" || echo "No changes"
          git push origin main || echo "Nothing to push"

      # 8. Build Docker Image
      - name: Build Docker image
        if: env.RUN_ID != ''
        run: |
          if [ -f "mlruns/*/${RUN_ID}/artifacts/model/MLmodel" ]; then
            mlflow models build-docker \
              --model-uri "runs:/${RUN_ID}/model" \
              --name creditscore_model

            echo "DOCKER_BUILT=true" >> $GITHUB_ENV
          else
            echo "Model not found. Skipping docker build."
            echo "DOCKER_BUILT=false" >> $GITHUB_ENV
          fi

      # 9. Login Docker Hub
      - name: Log in to Docker Hub
        if: env.DOCKER_BUILT == 'true'
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_HUB_USERNAME }}
          password: ${{ secrets.DOCKER_HUB_ACCESS_TOKEN }}

      # 10. Tag Docker image
      - name: Tag Docker image
        if: env.DOCKER_BUILT == 'true'
        run: |
          docker tag creditscore_model ${{ secrets.DOCKER_HUB_USERNAME }}/creditscore_model:latest

      # 11. Push Docker image
      - name: Push Docker image
        if: env.DOCKER_BUILT == 'true'
        run: |
          docker push ${{ secrets.DOCKER_HUB_USERNAME }}/creditscore_model:latest

      # 12. Done
      - name: Finished
        run: echo "✅ CI workflow done."