"""
__author__ = "Zhihua Liang"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""
import importlib
from deepspace.config.config import config, logger


def main():
    # Create the Agent then run it..
    agent_class = importlib.import_module('.agents', package='deepspace')
    agent_class = getattr(getattr(agent_class, config.settings.agent_module), config.settings.agent_name)
    agent = agent_class()
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()
