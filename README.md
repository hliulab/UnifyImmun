# UnifyImmun: a unified cross-attention model for prediction of antigen binding specificity 
UnifyImmun is an advanced computational model that predicts the binding specificity of antigens to both HLA and TCR molecules. By employing a unified cross-attention mechanism, UnifyImmun provides a comprehensive evaluation of antigen immunogenicity, which is crucial for the development of effective immunotherapies.

## Key features
- **Unified model**: Simultaneously predicts peptide bindings to both HLA and TCR molecules.
- **Cross-attention mechanism**: Integrates the features of peptides and HLA/TCR molecules for model interpretability.
- **Progressive training strategy**: Utilizes a two-phase progressive training to improve feature extraction and model generalizability.
- **Virtual adversarial training**: Enhances model robustness by training on perturbed data.
- **Superior performance**: Outperforms existing methods on both pHLA and pTCR prediction tasks on multiple datasets.

For inquiries or collaborations, please contact: hliu@njtech.edu.cn

## System requirements
- **Linux version**: 4.18.0-193 (Centos confirmed)
- **GPU**: NVIDIA GeForce RTX 4090 (or compatible GPUs)
- **CUDA Version**: 12.4
- **Python**: 3.10
- **PyTorch**: 2.2.1 (model implementation)

## Installation guide
>1. Clone the UnifyImmun repository

` git clone https://github.com/hliulab/unifyimmun.git`

>2. Enter UnifyImmun project folder

` cd unifyimmun/`

>3. Set up the Python environment and install the required packages
   
` pip install -r requirements.txt `

## Instructions for use
The training data for pHLA and pTCR bindings is stored in the <kbd>data</kbd> folder. The source code of UnifyImmun model, as well as the training and testing scripts, are included in the <kbd>source</kbd> folder. The trained models are stored in the <kbd>trained_model</kbd> folder.

### Input data format
The input data should be a CSV file with three columns named `tcr`, `peptide`, and `HLA`, representing the TCR CDR3 sequence, peptide sequence, and HLA sequence, respectively.

### Model training
For the convenience of sequentially running all the training steps, you can use the provided Python script named run_all_phases.py. After ensuring that the required environment and dependencies are installed, execute the following code:

`python ./source/run_all_phases.py`



### Model testing
Given your fine-tuned model or our trained model (saved in trained_model folder), you can evaluate it on our provided demo test set using the following test scripts.
>Predict HLA binding specificity using pHLA test set

`python ./source/HLA_test.py`

>Evaluate TCR binding specificity using pTCR test set

`python ./source/TCR_test.py`

> In our practice, the time overhead required to run the two demos above is about 2 minutes when batch_size=8192.


### Output pred score
You can use code to output the model pred scores, the result will be save in the <kbd>data</kbd> folder
>Output pHLA pred scores:

`python ./source/out_score_HLA.py --input ./data/data_HLA/independent_set  --output ./data/out_hla_independent`

>Output pTCR pred scores:

`python ./source/out_score_TCR.py --input ./data/data_TCR/independent_set  --output ./data/out_tcr_independent`



### Hyperparameter adjustment
If transfer the model using your custom dataset, you may need to adjust the hyperparameters within the Python scripts. Hyperparameters include learning rate, batch size, number of epochs, and other model-specific parameters.

Note: Ensure that the file paths and script names provided in the commands match those in your project directory. The source/ directory and script names like HLA_test.py and TCR_test.py are placeholders and should be replaced with the actual paths and filenames used in your implementation.

### Customizing output
To customize the output results, users can modify the parameters within each script. Detailed comments within the code provide descriptions and guidance for parameter adjustments.

## Support
For further assistance, bug reports, or to request new features, please contact us at hliu@njtech.edu.cn or open an issue on the [GitHub repository page](https://github.com/hliulab/unifyimmun).

---

Please replace the placeholder links and information with actual data when the repository is available. Ensure that the instructions are clear and that the repository contains the `requirements.txt` file with all necessary dependencies listed.
