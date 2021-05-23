from flask import Flask


class FlaskHook(Flask):
    """
    e.g.
    hook = FlaskHook()
    agent.hook = hook
    agent.hook.run()
    """
    def __init__(self):
        self._agent = None

    @property
    def agent(self):
        return self._agent

    @agent.setter
    def agent(self, val):
        self._agent = val
        self.add_url_rule('/act', 'act', self._agent.act)
