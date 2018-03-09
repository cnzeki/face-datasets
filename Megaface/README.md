# MegaFace

Some script for the [MegaFace](http://megaface.cs.washington.edu) Test.

# Alignment

## Facescrub

```
python align_facescrub.py facescrub_image_dir  aligned_dir  features_list_json_path
```

##  MegaFace

	python align_megaface.py megaface_image_dir  aligned_dir  distractor_templatelists_dir

# Extract feature(with PyCaffe)

## Setup

Copy the `config.conf.example` file `config.conf` and modify the paths to fit your environment.

##  Run

You can write more than one models in config.conf, and choose the model in the section `[model].name` field or specify it in command parameters.

	python extract_feature_list.py config.conf  [model_name]

