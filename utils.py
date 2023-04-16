import yaml
import os

def make_yaml(dest):
    train_yaml = dict(
        train =f'{dest}/train',
        val =f'{dest}/val',
        test=f'{dest}/test',
        nc =1,
        names =['ship',]
    )

    with open(os.path.join(dest,'train.yaml'), 'w') as outfile:
        yaml.dump(train_yaml, outfile, default_flow_style=True)



