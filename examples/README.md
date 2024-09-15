## Install

Make sure you've installed `fastdata` with the following command from the root of the repo:

```bash
pip install -e .
```

then if you will be training a model, install the following dependencies:

```bash
pip install -r requirements.txt
```

then run the following if you will use flash attention:

```bash
pip install flash-attn --no-build-isolation
```

## Run

### Data Synthesis

Right now we have a script for generating our tiny programs dataset, which can be run with the following command:

```bash
python tiny_programs.py
```

You can see all the command line arguments by running:

```bash
python tiny_programs.py --help
```

### Training

To train a model, you can use the following command:

```bash
python train.py
```

Similarly, you can see all the command line arguments by running:

```bash
python train.py --help
```