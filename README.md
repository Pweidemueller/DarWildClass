# DarWildClass

This project serves to classify animals detected through the wild cameras installed at Darwin College, Cambridge, UK.

Since we don't have labelled datasets from our cameras, we were hoping to just use already general-purpose pre-trained classification models.

The cameras only capture a video clip when they observe movement, hence identifying if there is an animal in the any given frame is not the main task/problem. As soon as movement is identified the camera records a 10s clip consisting of >300 frames.

Some issues identified:

- if camera points onto water, then there are clips that don't have animals in it, likely because the water movement triggered the camera
- locations where there is a stark contrast between sun and shade make it harder for the model to identify animals (or any objects)

## Environment

I'm using python, mainly with the `timm` package. Packages are installed in a conda environment `darwild`. All packages and versions can be found in this file: `XXX.yml`.

## Procedure

1. Get all frames in the 10s clip.
2. For each frame run the pre-trained `tf_efficientnet_b5.ns_jft_in1k` model from https://github.com/huggingface/pytorch-image-models (Artem's recommendation).
3. Get the top 5 predictions in each frame.
4. Count in how many frames each top 5 class was detected.

## Testing

### `DSCF0935.MP4`

There's two ducks (male and female) in the footage. However these are the classes the model finds:

```{python}
[{'label': 'lakeside, lakeshore', 'number of frames:': 307}, {'label': 'valley, vale', 'number of frames:': 263}, {'label': 'dam, dike, dyke', 'number of frames:': 159}, {'label': 'water_snake', 'number of frames:': 153}, {'label': 'canoe', 'number of frames:': 142}, {'label': 'beaver', 'number of frames:': 126}, {'label': 'cliff, drop, drop-off', 'number of frames:': 94}, {'label': 'drake', 'number of frames:': 71}, {'label': 'American_coot, marsh_hen, mud_hen, water_hen, Fulica_americana', 'number of frames:': 71}, {'label': 'goose', 'number of frames:': 70}, {'label': 'platypus, duckbill, duckbilled_platypus, duck-billed_platypus, Ornithorhynchus_anatinus', 'number of frames:': 39}, {'label': 'black_swan, Cygnus_atratus', 'number of frames:': 20}, {'label': 'alp', 'number of frames:': 10}, {'label': 'barracouta, snoek', 'number of frames:': 6}, {'label': 'tench, Tinca_tinca', 'number of frames:': 4}]
```

=> It identifies `drake` (male duck) in 71/307 frames. But there are several other animal classes it identfies more frequently.

### `DSCF0005.MP4`

There is a otter moving through the clip in the first 2 seconds. It's face is hardly visible.

```{python}
[{'label': 'badger', 'number of frames:': 306}, {'label': 'armadillo', 'number of frames:': 306}, {'label': 'grey_fox, gray_fox, Urocyon_cinereoargenteus', 'number of frames:': 304}, {'label': 'porcupine, hedgehog', 'number of frames:': 274}, {'label': 'skunk, polecat, wood_pussy', 'number of frames:': 273}, {'label': 'beaver', 'number of frames:': 25}, {'label': 'mink', 'number of frames:': 18}, {'label': 'wombat', 'number of frames:': 10}, {'label': 'wild_boar, boar, Sus_scrofa', 'number of frames:': 6}, {'label': 'otter', 'number of frames:': 2}, {'label': 'cougar, puma, catamount, mountain_lion, painter, panther, Felis_concolor', 'number of frames:': 2}, {'label': 'mongoose', 'number of frames:': 1}, {'label': 'weasel', 'number of frames:': 1}, {'label': 'jaguar, panther, Panthera_onca, Felis_onca', 'number of frames:': 1}, {'label': 'red_fox, Vulpes_vulpes', 'number of frames:': 1}]
```

The model finds related animals. Otter is found in only 2 frames.