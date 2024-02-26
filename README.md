# Vanila Pytorch + Hydra


Yet another Pytorch boilerplate ...... but targeting at research users who would love to 
- 1) write customize steps
- 2) do extensive ablation studies

### Key features

- Vanilla Pytorch with a trainer.
    - The coding style follows [HuggingFace-Transformers-trainer](https://huggingface.co/docs/transformers/main_classes/trainerm).
    - The trainer consists of main functions like `train`, `training_step`, `evaluation_step`, etc.
- [Hydra](https://hydra.cc/) supports [multi-run](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/), a command line configuration, and verbose logging of hyper-parameters for ablation studies.

- MISC 
    - Custom dataset.
    - Currently, it is a single-GPU implementation.
    - Coding style: one lesson I learned from my mentors is to avoid global variables; instead, pass the hyper-parameters down function by function, which is less prone to unnoticeable bugs.

### Samplary usage
- `python main.py`
- `python main.py --multirun model.arch=RNN,CNN`
(reference: conf/config.yaml)

This demo is built for an audio classification task:
- The `data/` folder comes with 6 short audio clips of piano or podcast. The samplary datasets classes can load the raw wav files and segment them into short training and testing samples. 
- It runs a vanilla RNN or CNN model that binary-classifies the two audio types.

### Future work
- Support multi-GPU training using DDP.


### License
Code released under [WTFPL](http://www.wtfpl.net/)

