import subprocess


def generate_unzip_commands(zip_min: int, zip_max: int):
    unzip_commands = []
    for i in range(zip_min, zip_max + 1):
        unzip_commands.append(f'unzip PadChest/{i}.zip -d PadChest/images')
        unzip_commands.append(f'rm PadChest/{i}.zip -f')
    return unzip_commands


if __name__ == '__main__':
    commands = generate_unzip_commands(6, 50)
    commands.extend(generate_unzip_commands(54, 54))

    for command in commands:
        print(f'starting: {command}')
        proc = subprocess.Popen(command.split())
        proc.wait()
        print(f'finished: {command}')