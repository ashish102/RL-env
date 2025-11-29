# Installation Guide for Debian Linux

Follow these steps to set up the project on a fresh Debian server.

## 1. Update System and Install Prerequisites
Ensure your system is up to date and has the necessary tools (curl, git).

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl git
```

## 2. Install uv (Package Manager)
`uv` manages Python versions and dependencies automatically. You do **not** need to manually install Python; `uv` will fetch the correct version (3.12) for you.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

## 3. Clone the Repository
Replace the URL with your repository URL.

```bash
git clone https://github.com/ashish102/RL-env.git
cd RL-env
```

## 4. Install Dependencies
Run `uv sync` to create the virtual environment. To run tests, you must include the `dev` extras.

```bash
uv sync --extra dev
```

## 5. Activate Environment
To use the installed packages, activate the environment:

```bash
source .venv/bin/activate
```

## 6. Verify Installation
Run the tests to make sure everything is working.

```bash
uv run pytest
```

## 7. View Coverage Report Remotely
To view the coverage report in your browser:

1.  **Generate the report** (if not already done):
    ```bash
    uv run pytest --cov=gym_pydantic --cov-report=html
    ```

2.  **Start a simple HTTP server**:
    ```bash
    cd htmlcov
    python3 -m http.server 8000 --bind 0.0.0.0
    ```

3.  **Access in Browser**:
    Open `http://<YOUR_EXTERNAL_IP>:8000` in your web browser.

    *Note: Ensure your GCP firewall allows TCP traffic on port 8000.*
