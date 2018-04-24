# codebythesea-tensorflow

## Requirements

* Python 3

### [macOS](https://www.tensorflow.org/install/install_mac)

#### Python

Steps require [Homebrew](https://brew.sh).

```bash
brew install python && brew upgrade python
python3 -m pip install -U pip setuptools wheel virtualenv
```

### [Ubuntu](https://www.tensorflow.org/install/install_linux)

#### Python

```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-dev
```

### [Windows](https://www.tensorflow.org/install/install_windows)

#### Python

Steps require [Chocolatey](https://chocolatey.org/).

```bash
choco install python3
```

## Usage

### Clone submodules

```bash
git submodule init && git submodule update --recursive --remote
```

### Install TensorFlow with virtualenv

```bash
virtualenv --system-site-packages -p python3 .venv
source ./.venv/bin/activate
pip3 install -U -r ./requirements.txt
```

### Start Tensorboard

```bash
tensorboard --logdir=./logs
```

## Model

[Download](https://drive.google.com/drive/folders/1sNl2_bu-iNKbUyB0lBi55-SygrPwJK8p?usp=sharing) working model.

## Resources

1. [Tensorflow Tutorial 2: image classifier using convolutional neural network](http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/)
2. [Jian Yang: hotdog identifying app](https://www.youtube.com/watch?v=vIci3C4JkL0
)
3. [Hotdog-Classification images](https://github.com/hayzamjs/Hotdog-Classification/tree/master/images
)
4. [Installing TensorFlow
](https://www.tensorflow.org/install/)
