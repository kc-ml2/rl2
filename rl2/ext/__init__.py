import os
import requests
import docker


class DockerFlaskClient:
    def __init__(self, image, port=5000):
        self.image = image
        self.client = docker.from_env()
        # models loading and flask entrypoint are ran in docker entrypoing/command
        if self.client.images.get(image):
            # will raise error and not instantiate if image not exists
            self.container = self.client.containers.run(image)

        self.end_point = 'http://' + str(image) + '/act:' + str(port)

    def act(self, obs):
        action = requests.post(self.end_point, obs)

        return action
