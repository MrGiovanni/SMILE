# SMILE Inference Detailed Guide Book
### step 1. download SMILE models checkpoints and demo dataset
To inference SMILE, please first download the pre-trained checkpoints for VAE, Classifier and Segmenter, as well as the trained SMILE model.

[![HuggingFace](https://img.shields.io/badge/HuggingFace-SMILE_Model-yellow.svg?logo=huggingface)](https://huggingface.co/MitakaKuma/SMILE)

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Demo_Dataset-yellow.svg?logo=huggingface)](https://huggingface.co/MitakaKuma/SMILE_Demo_Dataset)
```bash
bash download_ckpts.sh
bash download_smile_demo_data.sh
```

<details> 
<summary>Click to download deatiled model checkpoints 🤖</summary>
<br>

All of the pre-trained will be download to `./ckpt` folder.
## Download the Super-VAE checkpoints
```bash
hf download MitakaKuma/SMILE --include="autoencoder/*" --local-dir "./ckpt" 
```

## Download the Classification model:
```bash
hf download MitakaKuma/SMILE --include="classifier/*" --local-dir="./ckpt"
```

## Download the Segmentation model:
```bash
hf download MitakaKuma/SMILE --include="segmenter/*" --local-dir="./ckpt"
```

## Download the SMILE checkpoints
```bash
hf download MitakaKuma/SMILE --include="SMILE/*" --local-dir="./ckpt"
```

</details>

### step 2. 🍳 prepare the data (🔔 if with the demo dataset, skip):

The data for inference is expected to be organized into BDMAP form, as below:

```bash
  DATA_FOLDER
    DATASET_NAME/
    ├── BDMAP_00012345/
    │     └── ct.nii.gz
    ├── BDMAP_00067890/
    │     └── ct.nii.gz
```

<a id="multi-ct-inference"></a>

### step 3. 🍇 choose inference mode:
1. **dataset inference (PREFERRED, general purpose):**
    ```bash
    cd SMILEModel
    bash inference_multiple_CT.sh
    ```
    <details> <summary>Click to view Configuration Details</summary>

    1. Edit the User Defined section in the script to point to your data, model, and guide CSV:
          ```bash
          # ================ User Defined ================ #
          MODEL_CKPT_NAME="SMILE/Version (default as v0.2)"
          DATA_FOLDER="/path/to/nifti/dataset/parent/folder"
          DATASET_NAME="/dataset/name/as/nnunetv2/from"
          # Enhancement targets
          TARGETS=("arterial" "venous" "delayed")
          GUIDE_CSV="/nifti/you/want/to/inference"
          # ============================================== #
          ```

        `MODEL_CKPT_NAME`: Version of SMILE model, deault as v0.2 (stable and fast).

        `DARA_FOLDER`: Parental folder to the inference dataset.

        `DATASET_NAME`: Name of the inference dataset, desired in *nnunetv2* form.

        `TARGETS`: One or multiple enhancement targets (comma-separated).  
          Options: `non-contrast`, `arterial`, `venous`, `delayed`.

        `GUIDE_CSV (optional)`: .csv file to guide model to inference on specific cases. An example:<br>
        ```
        Inference ID
        BDMAP_xxxx01
        BDMAP_xxxx02
        BDMAP_xxxx03
        BDMAP_xxxx04
        ...
        ```


    2. The outputs are supposed to be stored in:
        ```bash
        SMILE/out/DATASETNAME-SMILE_Version
        # example: ../out/Dataset101_Example-SMILE_v0.2
        ```

    </details>


2. **single-CT case infernce (ONLY for testing and debuging):**
    ```bash
    cd SMILEModel
    bash inference.sh \
        --gpu_id 0 \
        --source non-contrast \
        --target arterial,venous,delayed \
        --patient_id Example-1
    ```

    <details> <summary>Click to view Configuration Details</summary>

      1. Edit the argument section when running the script to specify the GPU, source phase, target phases, patient ID, and model:

          ```bash
              --gpu_id 0 \
              --source non-contrast \
              --target arterial,venous,delayed \
              --patient_id Example-1
          ```
      - **`--gpu_id`**: GPU index used for inference (default = `0` if omitted).

      - **`--source`**: Input CT phase.  
          Options: `non-contrast`, `arterial`, `venous`, `delayed`.

      - **`--target`**: One or multiple enhancement targets (comma-separated).  
          Options: `non-contrast`, `arterial`, `venous`, `delayed`.

      - **`--patient_id`**: Patient identifier following BDMAP naming (e.g., `RS-GIST-121`).

      - **`--model`**: SMILE model version to load.  
          Default: `SMILE_v0.2.2-a100-200k`.

      2. The script automatically finds the correct input NIfTI file, loads the fine-tuned UNet + VAE, and performs multi-phase translation in a single call.

      3. The outputs are supposed to be stored in:
          ```bash
          SMILE/out/DATASETNAME-SMILE_Version
          # example: ../out/Dataset101_Example-SMILE_v0.2
          ```
    </details>

