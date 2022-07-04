GP_TEMPLATE = """
datasets:  !include configs/gp/{name}/{epochs}/datasets.yaml
model:     !include configs/gp/models/{model}.yaml
optimizer: !include configs/gp/optimizer.yaml
train:     !include configs/gp/{name}/{epochs}/train.yaml
test:      !include configs/gp/{name}/test.yaml
""".lstrip()

IMAGE_TEMPLATE = """
datasets:  !include configs/image/{name}/datasets.yaml
model:     !include configs/image/models/{model}.yaml
optimizer: !include configs/image/optimizer.yaml
train:     !include configs/image/{name}/train.yaml
test:      !include configs/image/{name}/test.yaml
""".lstrip()

SIM2REAL_TEMPLATE = """
datasets:  !include configs/sim2real/{name}/datasets.yaml
model:     !include configs/sim2real/models/{model}.yaml
optimizer: !include configs/sim2real/optimizer.yaml
train:     !include configs/sim2real/{name}/train.yaml
test:      !include configs/sim2real/{name}/test.yaml
""".lstrip()

GP_PATH = "configs/gp/{name}/{epochs}/{model}.yaml"
IMAGE_PATH = "configs/image/{name}/{model}.yaml"
SIM2REAL_PATH = "configs/sim2real/{name}/{model}.yaml"


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    args = parser.parse_args()

    print("Add following files yourself:")
    print(f"- configs/gp/models/{args.model}.yaml")
    print(f"- configs/image/models/{args.model}.yaml")
    print(f"- configs/sim2real/models/{args.model}.yaml")
    print()

    print("Creating config files...")

    for name in ["matern", "periodic", "rbf", "t_noise"]:
        for epochs in ["100", "125", "250", "500", "inf"]:
            filename = GP_PATH.format(name=name, epochs=epochs, model=args.model)
            print(f"Create {filename}")
            with open(filename, "w") as f:
                f.write(GP_TEMPLATE.format(name=name, epochs=epochs, model=args.model))

    for name in ["celeba", "mnist", "svhn"]:
        filename = IMAGE_PATH.format(name=name, model=args.model)
        print(f"Create {filename}")
        with open(filename, "w") as f:
            f.write(IMAGE_TEMPLATE.format(name=name, model=args.model))

    for name in ["lotka_volterra"]:
        filename = SIM2REAL_PATH.format(name=name, model=args.model)
        print(f"Create {filename}")
        with open(filename, "w") as f:
            f.write(SIM2REAL_TEMPLATE.format(name=name, model=args.model))

    print("Done")
