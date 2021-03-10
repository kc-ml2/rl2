import os
import requests
import docker

class DockerFlaskClient:
    def __init__(self, image, port=5000):
        self.image = image
        self.cx = docker.from_env()
        # models loading and flask entrypoint are ran in docker entrypoing/command
        self.container = self.cx.containers.run(image)
        self.end_point = 'http://' + str(image) + '/act:' + str(port)

    def act(self, obs):
        action = requests.post(self.end_point, obs)

        return action