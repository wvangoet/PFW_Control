#!/usr/bin/env python
from setuptools import setup, find_packages
from pathlib import Path

HERE= Path(__file__).parent.absolute()
with (HERE / 'README.md').open('rt', encoding='utf-8') as fh:
	LONG_DISCRIPTION = fh.read().strip()

REQUIREMENTS: dict = {
	"core": [
		"numpy >= 1.0",
		"matplotlib >= 3.0",
		"tensorflow >= 2.6",
		"pandas >= 1.0",
	],
}

setup(
	name="mlp-pfw-control",
	version="0.0.1",
	author="Wietse Van Goethem",
	author_mail="wietse.van.goethem@cern.ch",
	description="first attempt of an upgraded NN for pfw control",
	long_description='test',
	long_description_content_type="text/markdown",
	packages=find_packages(),
	python_requires=">=3.6, <4",
	classifiers=[
		"Programming Language :: Python :: 3",
		"Intended Audience :: Science/Research",
		"Operating System :: OS Independent",
	    ],
	package_data={'':
		['mlp-models.toml'],
	},
	install_requirements=REQUIREMENTS['core'],
)
