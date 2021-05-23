import docker

from server import DockerFlaskClient

# container = DockerFlaskClient('anthonyjung/randomagent:v5')

client = docker.from_env()
# models loading and flask entrypoint are ran in docker entrypoing/command
image = 'anthonyjung/randomagent:v5'
print(client.images.get(image))

# will raise error and not instantiate if image not exists
container = client.containers.run(image, detach=True, auto_remove=True)

# print(container)