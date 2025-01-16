from setuptools import find_packages, setup

with open("./requirements.txt", "r") as f:
    requirements = [l.strip() for l in f.readlines() if len(l.strip()) > 0]

setup(
    name="dfpe",
    version="2.1.0",
    packages=find_packages(),
    install_requires=requirements,
    author="Enrique Solarte",
    description=("Direct Floor Plan Estimation proposed in ICRA23 / RA-L"),
    license="BSD",
)
