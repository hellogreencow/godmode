#!/usr/bin/env python3
"""
Setup script for GodMode CLI
"""

from setuptools import setup, find_packages

setup(
    name="godmode-cli",
    version="1.0.0",
    description="Simple CLI for GodMode Hierarchical Reasoning System",
    py_modules=["godmode_cli"],
    install_requires=[
        "click>=8.0.0",
        "rich>=13.0.0",
    ],
    entry_points={
        "console_scripts": [
            "godmode=godmode_cli:godmode",
        ],
    },
    python_requires=">=3.11",
)