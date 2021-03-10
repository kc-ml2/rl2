import docker

class DockerAgent:
    def __init__(self, image):
        self.image = image
        self.cx = docker.from_env()

    def act(self):
        self.cx.containers.run(self.image, 'act')
