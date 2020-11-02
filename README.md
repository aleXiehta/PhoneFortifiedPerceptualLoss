# Phone Fortified Perceptual Loss for Speech Enhancement
This is the official implementation of our paper *"Improving Perceptual Quality by Phone-Fortified Perceptual Loss for Speech" Enhancement"*

### Requirements
- pytorch 1.6
- torchcontrib 0.0.2
- torchaudio 0.6.0
- pesq 0.0.1
- colorama 0.4.3
- fairseq 0.9.0

### Data preparation
##### Enhancement model parameters and the *wav2vec* pre-trained model
Please download the model weights from [here](https://drive.google.com/file/d/1QP2bcmnn1yHybsmUbCj9f0xjUyRvrqJa/view?usp=sharing), and put the weight file into the `checkpoint` folder.
The *wav2vec* pre-trained model can be found in the official [repo](https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md#pre-trained-models-1).

##### Voice Bank--Demand Dataset
The Voice Bank--Demand Dataset is not provided by this repository. Please download the dataset and build your own PyTorch dataloader from [here](https://datashare.is.ed.ac.uk/handle/10283/1942?show=full).
For each `.wav` file, you need to first convert it into 16kHz format by any audio converter (e.g., [sox](http://sox.sourceforge.net/)).
```
sox <48K.wav> -r 16000 -c 1 -b 16 <16k.wav>
```

### Usage
##### Training
To train the model, please run the following script.
The full training process apporximately consumes 19GB of GPU vram. Reduce the batch size if needed.
```
python main.py \
    --exp_dir <root/dir/of/experiment> \
    --exp_name <name_of_the_experiment> \
    --data_dir <root/dir/of/dataset> \
    --num_workers 16 \
    --cuda \
    --log_interval 100 \
    --batch_size 28 \
    --learning_rate 0.0001 \
    --num_epochs 100 \
    --clip_grad_norm_val 0 \
    --grad_accumulate_batches 1 \
    --n_fft 512 \
    --hop_length 128 \
    --model_type wav2vec \
    --log_grad_norm
```
##### Testing
To generate the enhanced sound files, please run:
```
python generate.py <path/to/your/checkpoint/ckpt> <path/to/output/dir>
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
* [Bio-ASP Lab](https://bio-asplab.citi.sinica.edu.tw), CITI, Academia Sinica, Taipei, Taiwan
