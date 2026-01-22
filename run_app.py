#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# run_app.py — single-command launcher for Streamlit REM app
# Suppresses most of Streamlit/Tornado log spam (WebSocketClosedError, etc.)

import subprocess
import os
import sys
import logging

# Path to app.py (this file should live in the same folder)

app = os.path.join(os.path.dirname(__file__), "app.py")

# Quiet down warnings globally

os.environ.setdefault("PYTHONWARNINGS", "ignore")

# Suppress noisy Tornado loggers

for name in (
    "tornado.general",
    "tornado.access",
    "tornado.application",
    "tornado.websocket",
):
    logging.getLogger(name).setLevel(logging.ERROR)

# Run Streamlit with reduced logging

subprocess.run([
    sys.executable, "-m", "streamlit", "run", app,
    "--logger.level=error"
])
