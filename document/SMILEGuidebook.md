## SMILE Guide Book
### step 0. download SMILE models checkpoints
To inference SMILE, please first download the pre-trained checkpoints for VAE, Classifier and Segmenter, as well as the trained SMILE model.

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow.svg?logo=huggingface)](https://huggingface.co/your-model-link)

<details> 
<summary>download model checkpoints:</summary>

## Download the Super-VAE checkpoints
```bash
hf download CVPR-SMILE/SMILE_mini --include="autoencoder/*" --local-dir "./ckpt" 
```

## Download the Classification model:
```bash
hf download CVPR-SMILE/SMILE_mini --include="classifier/*" --local-dir="./ckpt"
```

## Download the Segmentation model:
```bash
hf download CVPR-SMILE/SMILE_mini --include="segmenter/*" --local-dir="./ckpt"
```

## Download the SMILE checkpoints
```bash
hf download CVPR-SMILE/SMILE_mini --include="SMILE_v0.2/*" --local-dir="./ckpt"
```

</details>

### step 1. prepare the data:

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
### step 2. choose inference mode:
1. **multiple-CT infernce / dataset inference:**
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

        `GUIDE_CSV`: Case list CSV, containing the cases to inference.


    2. The outputs are supposed to be stored in:
        ```bash
        SMILE/out/DATASETNAME-SMILE_Version
        # example: ../out/Dataset101_Example-SMILE_v0.2
        ```

    </details>


2. **single-CT case infernce:**
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

