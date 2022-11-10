# ConNER
We present **ConNER** (Enhancing Label **Con**sistency on Document-level **N**amed **E**ntity **R**ecognition), a method that improves label agreement of modifiers (e.g., adjectives and prepositions) to make your models more consistent on biomedical text. This repository provides a way to train and evaluate our <em>ConNER</em> approach. Please see our [paper](https://arxiv.org/abs/2210.12949) for more details.

<div align="center">
      <img src="docs/images/conner_structure.PNG" width="95%">
</div>

## Updates

* \[**Oct 12, 2022**\] First code updates.

## Quick Link
* [Installation](#installation)
* [Resources: datasets, pre-trained models, fine-tuned models](#resources)
* [Traning and Inference](#training-and-inference)
* [References](#references)
* [License](#license)
* [Contact Info.](#contact-information)

## Installation
You need to install dependencies to use ConNER.
```bash
# Install torch with conda (please check your CUDA version)
conda create -n conner python=3.8
conda activate conner
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install ConNER 
git clone https://github.com/dmis-lab/ConNER.git
pip install -r requirements.txt
```

## Resources

### 1. Datasets
We updated our resource files of four biomedical benchmarks, pre-trained model, and fine-tuned models. 
#### Datasets
* [NCBI-disease](http://nlp.dmis.korea.edu/projects/conner-jeong-et-al-2022/data/ncbi-disease.tar.gz)
* [CDR](http://nlp.dmis.korea.edu/projects/conner-jeong-et-al-2022/data/bc5cdr.tar.gz)
* [AnatEM](http://nlp.dmis.korea.edu/projects/conner-jeong-et-al-2022/data/anatem.tar.gz)
* [Gellus](http://nlp.dmis.korea.edu/projects/conner-jeong-et-al-2022/data/gellus.tar.gz)

#### Fine-tuned
* [NCBI-disease](http://nlp.dmis.korea.edu/projects/conner-jeong-et-al-2022/fine-tuned/ncbi-disease.tar.gz)
* [CDR](http://nlp.dmis.korea.edu/projects/conner-jeong-et-al-2022/fine-tuned/bc5cdr.tar.gz)
* [AnatEM](http://nlp.dmis.korea.edu/projects/conner-jeong-et-al-2022/fine-tuned/anatem.tar.gz)
* [Gellus](http://nlp.dmis.korea.edu/projects/conner-jeong-et-al-2022/fine-tuned/gellus.tar.gz)

#### Pre-trained Models (BioBERT or BioLM)
```bash
wget https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-Voc-hf.tar.gz
tar -zxvf RoBERTa-large-PM-M3-Voc-hf.tar.gz
```

## Training and Inference
```bash
make run-ner DATA_TYPE=doc_cons DATA_NAME=ncbi-disease MODEL_TYPE=roberta DATA_DIR='./data/' EPOCH=50 LR=3e-5 SEED=1 run_name=check MODEL_NAME=/directory/of/BioLM LM=biolm-large 
```

## References
Please cite our paper if you use ConNER in your work:
```bash
@article{jeong2022enhancing,
  title={Enhancing Label Consistency on Document-level Named Entity Recognition},
  author={Jeong, Minbyul and Kang, Jaewoo},
  journal={arXiv preprint arXiv:2210.12949},
  year={2022}
}
```

## License
Please see LICENSE for details.

## Contact Information
Please contact Minbyul Jeong (`minbyuljeong (at) korea.ac.kr`) for help or issues using ConNER.
