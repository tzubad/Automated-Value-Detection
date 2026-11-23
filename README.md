# Automated Value Detection in TikTok Content from Youth-Targeting Influencers

Code and data accompanying the paper *"Automated Value Detection in Tiktok Content from Youth-Targeting Influencers: A Large Language Model Approach"* (Starovolsky-Shitrit, Neduva, Appel-Doron, Daniel, Gafni, Tsur; JAIR, AI and Society track).

We annotated 885 TikToks from 91 youth-targeting influencers for the 19 values in Schwartz's theory of personal values (endorsed / confronted). We then compared two pipelines for detecting these values automatically:

- **Direct (d):** an LLM extracts values directly from the mp4 video.
- **2-step (2s):** an LLM first converts the video into a textual script (scene heading, action, dialogue). Values are then extracted from the script by an LLM (few-shot) or by a fine-tuned masked language model (MLM).

## Repository layout

```
data/
  annotations/
    ground_truth_labels.csv        the 19 value labels per TikTok used as ground truth
  predictions/
    raw/                           unaggregated model outputs, one row per TikTok
      gemini_direct_video2values.csv     Gemini^d   (video -> values)
      gemini_2step_script2values.csv     Gemini^2s  (script -> values)
      mistral_2step_script2values.csv    Mistral^2s (script -> values)
    cleaned/                       raw outputs after the cleaning step in notebooks/4_evaluation
    mlm/                           per-fold predictions of the 3-fold CV MLM classifiers
      roberta_large/                         RoBERTa^2s       fold{0,1,2}_predictions.csv
      deberta_v2_xlarge/                     DeBERTa^2s
      roberta_large_continued_pretraining/   RoBERTa^2s_PT
      deberta_v2_xlarge_continued_pretraining/ DeBERTa^2s_PT
prompts/                           exact LLM system instructions (see prompts/README.md)
  step1_video_to_script.txt        video -> script (2-step, step 1)
  direct_video_to_values.txt       video -> values (direct)
  step2_script_to_values.txt       script -> values (2-step, step 2)
notebooks/
  1_video_to_script/               step 1 of the 2-step pipeline (Gemini 1.5 Pro)
  2_llm_value_extraction/          LLM few-shot value extraction (direct and 2-step)
  3_mlm_classification/            RoBERTa / DeBERTa classifiers on scripts (3-fold CV)
  4_evaluation/                    cleaning, F1 computation, ground-truth co-occurrence
```

### Notebook-to-results map (Table 3 in the paper)

| Paper row | Notebook(s) |
|---|---|
| Gemini^d | `2_llm_value_extraction/gemini_direct_video2values.ipynb` → `4_evaluation/evaluate_direct_video2values.ipynb` |
| Gemini^2s | `1_video_to_script/gemini_video_to_script.ipynb` → `2_llm_value_extraction/gemini_2step_script2values.ipynb` → `4_evaluation/evaluate_script2values.ipynb` |
| Mistral^2s | scripts from step 1 → `2_llm_value_extraction/mistral_2step_script2values.ipynb` → `4_evaluation/evaluate_script2values.ipynb` |
| RoBERTa^2s, DeBERTa^2s | `3_mlm_classification/roberta_large.ipynb`, `deberta_v2_xlarge.ipynb` |
| RoBERTa^2s_PT, DeBERTa^2s_PT | `3_mlm_classification/*_continued_pretraining.ipynb` |
| Figures 1–2 (label statistics) | `4_evaluation/ground_truth_cooccurrence.ipynb` |

## Data

### Label encoding

`ground_truth_labels.csv` and all prediction files have one column per Schwartz value (`1_Value1_<Value>_values`). Each cell holds one of:

| Cell value | Meaning | Binary label used in evaluation |
|---|---|---|
| `present` | value endorsed | endorsed = 1 |
| `dominant` | value endorsed and dominant | merged into endorsed = 1 (see paper §3.2) |
| `conflict` | value confronted | confronted = 1 |
| empty | value absent | 0 / 0 |

Each TikTok therefore maps to a vector of 38 binary labels (19 endorsed and 19 confronted).

- **Identifiers:** in the ground truth, `1_Link1` is the public TikTok URL. In the prediction files, `filename` (or `1_Link1` in the cleaned files) is `@<handle>_video_<video id>`, which joins with the URL.
- **Extra columns in the raw prediction files:** `Has_sound` and `Additional Notes`, both returned by the model.

### MLM fold predictions

Each `data/predictions/mlm/<model>/fold<k>_predictions.csv` has one row per TikTok (869 videos, all of them in the ground truth). Its columns are:

| Column | Meaning |
|---|---|
| `original_path` | TikTok URL |
| `path` | script/video file name (`@<handle>_video_<id>.mp4`) |
| `split` | the role of the video in fold *k*: `train`, `test` or `none` |
| `pred_<VALUE>` | binary prediction for the endorsed value |
| `pred_<VALUE>_NEG` | binary prediction for the confronted value |

- **Value codes:** SDT / SDA = self-direction thought/action, ST = stimulation, HE = hedonism, AC = achievement, POR / POD = power resources/dominance, FA = face, SEP / SES = security personal/societal, COR / COI = conformity rules/interpersonal, TR = tradition, HU = humility, BED / BEC = benevolence dependability/caring, UNC / UNN / UT = universalism concern/nature/tolerance.
- **The `none` split:** 148 videos have `split = none` and are not used for training or testing in any fold. Across the 3 folds, each of the remaining 721 videos is in the test split exactly once.
- **Results:** reported results are computed on the `test` rows of each fold and averaged over folds ("best-epoch averaging", see paper §4.2).

### Collection and annotation

See paper §3 and Appendix A. In brief:
- Influencers were identified through surveys of children and adolescents (IRB-approved), LLM suggestions and manual verification.
- Up to 10 TikToks were sampled per influencer, using only publicly available content.
- Each TikTok was annotated independently by two trained annotators, and a values expert resolved disagreements (Gwet's AC1 = 0.77).

**Videos and generated scripts are not redistributed** (TikTok terms of service and creator privacy). The videos can be retrieved from the URLs in `ground_truth_labels.csv`.

## Running the code

The experiments were run on **Google Colab Enterprise**: a Compute Engine VM with 4 vCPUs, 16 GB RAM and an NVIDIA Tesla A100 (a2-highgpu-1g), in early 2025. The notebooks assume that environment:

- **Video and script storage:** the notebooks read videos and scripts from a Google Cloud Storage bucket. Set `project_id` and `bucket_name` to your own.
- **Gemini:** called through Vertex AI (`vertexai`, model `gemini-1.5-pro-002`). Requires a GCP project with Vertex AI enabled.
- **Mistral:** called through the `mistralai` SDK. Set the `MISTRAL_API_KEY` environment variable. The run used a hosted Mistral agent configured with the few-shot prompt in Appendix C (base model Mistral 7B).
- **MLMs:** `transformers` + `torch`, with `roberta-large` and `microsoft/deberta-v2-xlarge` from the Hugging Face Hub.
  - 3-fold CV via `sklearn.model_selection.KFold(n_splits=3, shuffle=True, random_state=42)`.
  - Early stopping with patience 3, batch size 8 (RoBERTa) / 4 (DeBERTa).
  - Best-epoch averaging across folds.
  - The continued-pretraining variants load checkpoints pretrained on the scripts.
- **Evaluation:** `pandas`, `numpy`, `matplotlib`, `seaborn`. The evaluation notebooks run as-is against the files in `data/`.
- **Plots:** the radar plots and bar charts in the paper were produced in R 4.4.2 (`fmsb`, `plotrix`, `ggplot2`, `dplyr`).

LLM calls are single runs per TikTok. The LLM extraction itself involves no randomness under our control.

## License

- **Code** (`notebooks/`): MIT, see [`LICENSE`](LICENSE).
- **Data** (`data/`): Creative Commons Attribution 4.0 International (CC BY 4.0), see [`data/LICENSE`](data/LICENSE).

## Citation

```bibtex
@article{starovolsky2026automated,
  title   = {Automated Value Detection in Tiktok Content from Youth-Targeting Influencers: A Large Language Model Approach},
  author  = {Starovolsky-Shitrit, Alina and Neduva, Alon and Appel-Doron, Naama and Daniel, Ella and Gafni, Itamar and Tsur, Oren},
  journal = {Journal of Artificial Intelligence Research},
  year    = {2026}
}
```
