from setuptools import setup, find_packages

setup(
	name="ExoRim",
	version="0.1",
	description="Module d'entrainment pour ExoRim. Sert a la reconstruction"
		"d'image a partir de la phase de Kerr.",
	packages=find_packages(exclude=("tests",)),
	python_requires=">=3.6"
)
