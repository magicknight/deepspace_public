"""
__author__ = "Zhihua Liang"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""
import importlib
from commontools.setup import config, logger


def main():
    # Create the Agent then run it..
    mod_name, func_name = config.deepspace.agent.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    agent = getattr(mod, func_name)()
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()
