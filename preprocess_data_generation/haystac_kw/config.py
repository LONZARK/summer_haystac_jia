import os, sys

assert os.environ["SUMO_HOME"] is not None, "SUMO_HOME not found; please check SUMO installation path"

SUMO_HOME = os.environ["SUMO_HOME"]

sys.path.insert(0, SUMO_HOME)

