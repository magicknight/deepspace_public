"""
__author__ = "Zhihua Liang"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""
import importlib
from deepsky.config import config, logger


def main():
    # Create the Agent then run it..
    agent_class = importlib.import_module(config.agent, package=None)
    agent = agent_class(config)
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()
