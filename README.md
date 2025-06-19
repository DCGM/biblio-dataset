# 📄 BiblioPage Evaluation Scripts

This repository contains official evaluation scripts for the **BiblioPage** dataset — a collection of 2,118 scanned title pages annotated with structured bibliographic metadata. The dataset serves as a benchmark for bibliographic metadata extraction, document understanding, and visual language model evaluation.

## 🔍 About the Dataset

- 2,118 annotated title pages from Czech libraries (1485–21st century)
- 16 bibliographic attributes (e.g., Title, Author, Publication Date, etc.)
- Annotations include both text and precise bounding boxes
- Evaluation results available for YOLO, DETR, and VLLMs (GPT-4o, LLaMA 3)

You can download the dataset on [Zenodo](https://zenodo.org/records/15683417).  
For more details, see the [BiblioPage paper](https://arxiv.org/abs/2503.19658v1).


## ✅ Evaluation Script

This repo provides:
- Attribute-level evaluation with precision, recall, F1, and mAP
- CER-based string matching
- Normalization for punctuation, whitespace, and historical characters
- JSON-to-JSON comparison for model predictions vs. ground truth


### 🔧 Usage

```bash
python evaluate_biblio_dataset.py \
  --record-dir <GROUND_TRUTH_DIR> \
  --result-dir <PREDICTION_DIR> \
  --normalize-record \
  --normalize-result \
  --confidence-threshold <CONFIDENCE_THRESHOLD> \
  [--show-prc-plot]
```

**Arguments**:

* `--record-dir <GROUND_TRUTH_DIR>`: Path to the directory containing the ground truth records (`labels/test`), as provided in the dataset archive downloadable from [Zenodo](https://zenodo.org/records/15683417).
* `--result-dir <PREDICTION_DIR>`: Path to the directory containing your model's predictions to be evaluated. Files must follow the expected format described below.
* `--normalize-record`: Apply normalization (punctuation, whitespace, historical characters) to ground truth records.
* `--normalize-result`: Apply normalization to predicted records.
* `--confidence-threshold <CONFIDENCE_THRESHOLD>`: Confidence level at which precision, recall, and F1 score are reported. Predictions below this threshold are ignored.
* `--show-prc-plot` *(optional)*: Display the Precision–Recall curve across all confidence thresholds.
* `--show-f1-plot` *(optional)*: Display the F1 score curve across all confidence thresholds.

> 💡 *In the BiblioPage paper, we selected the confidence threshold that yielded the highest F1 score. This optimal threshold is automatically printed to stdout when the script is run. You can run the evaluation once to retrieve the suggested threshold, then rerun it with `--confidence-threshold` set to that value for final results.*



### 📁 Expected Format of Predictions

Each prediction file (JSON) must follow the structure defined by the `BiblioResult` schema:

```python
class BiblioResult(BaseModel):
    task_id: Optional[str] = None
    library_id: str

    title: Optional[List[Tuple[str, float]]] = None
    subTitle: Optional[List[Tuple[str, float]]] = None
    partName: Optional[List[Tuple[str, float]]] = None
    partNumber: Optional[List[Tuple[str, float]]] = None
    seriesName: Optional[List[Tuple[str, float]]] = None
    seriesNumber: Optional[List[Tuple[str, float]]] = None
    edition: Optional[List[Tuple[str, float]]] = None
    placeTerm: Optional[List[Tuple[str, float]]] = None
    dateIssued: Optional[List[Tuple[str, float]]] = None
    publisher: Optional[List[Tuple[str, float]]] = None
    manufacturePublisher: Optional[List[Tuple[str, float]]] = None
    manufacturePlaceTerm: Optional[List[Tuple[str, float]]] = None
    author: Optional[List[Tuple[str, float]]] = None
    illustrator: Optional[List[Tuple[str, float]]] = None
    translator: Optional[List[Tuple[str, float]]] = None
    editor: Optional[List[Tuple[str, float]]] = None
```

Each bibliographic attribute should be a list of tuples:

* The first element is the predicted string value.
* The second is the confidence score (`float` between `0.0` and `1.0`).
  If your model does not provide confidence scores, set the value to `1.0`.

**Example**:

```json
{
  "library_id": "mzk.6b9b3cb0-4b86-11ed-9b54-5ef3fc9bb22f.3294f2eb-2f2a-4af8-ad2f-37fa7412e875",
  "title": [["JULIAN APOSTATA.", 0.94]],
  "subTitle": [["TRAGEDIE O 5 JEDNÁNÍCH.", 0.95], ["5 JEDNÁNÍCH.", 0.0001], ["TRAGEDIE O 5", 0.0001]],
  "seriesName": [["DRAMATICKÁ DÍLA JAROSLAVA VRCHLICKÉHO.", 0.96], ["DRAMATICKÁ DÍLA JAROSLAVA", 0.0002]],
  "seriesNumber": [["IX.", 0.61], ["IX.", 0.0003]],
  "placeTerm": [["PRAZE.", 0.87]],
  "dateIssued": [["1888.", 0.86]],
  "publisher": [["F. ŠIMÁČEK", 0.94]],
  "manufacturePublisher": [["F. ŠIMÁČEK", 0.17]]
}
```

### 🧪 Example Run & Output

```bash
python evaluate_biblio_dataset.py \
  --record-dir labels/test/ \
  --result-dir model_results/ \
  --normalize-record \
  --normalize-result 
```

**Example Output:**

```
410 biblio records loaded from labels/test/
410 biblio results loaded from model_results/

Normalizing record records...
Normalization done

Normalizing result records...
Normalization done
Evaluator: CER-01

                      Key                  GT   TP    FP   FN   Recall   Precision  F1    AP
-----------------------------------------------------------------------------------------------
                    title                 409  310   131   99   0.76     0.70       0.73  0.64
                 subTitle                 152   82    64   70   0.54     0.56       0.55  0.45
                 partName                  67   25    17   42   0.37     0.60       0.46  0.42
               partNumber                  93   51    20   42   0.55     0.72       0.62  0.54
               seriesName                  99   62    32   37   0.63     0.66       0.64  0.56
             seriesNumber                  89   42     7   47   0.47     0.86       0.61  0.58
                  edition                  83   50    10   33   0.60     0.83       0.70  0.65
                publisher                 381  206   125  175   0.54     0.62       0.58  0.53
                placeTerm                 368  313    58   55   0.85     0.84       0.85  0.85
               dateIssued                 319  266    23   53   0.83     0.92       0.88  0.85
     manufacturePublisher                 121   45    29   76   0.37     0.61       0.46  0.37
     manufacturePlaceTerm                 69    32     4   37   0.46     0.89       0.61  0.57
                   author                 336  245   100   91   0.73     0.71       0.72  0.68
              illustrator                  68   13     5   55   0.19     0.72       0.30  0.25
               translator                  63   24    11   39   0.38     0.69       0.49  0.37
                   editor                  67   10    23   57   0.15     0.30       0.20  0.07
-----------------------------------------------------------------------------------------------
                      AVG                 N/A  N/A   N/A  N/A   0.53     0.70       0.59  0.52

Best confidence threshold: 0.30
```

> 💡 *As shown above, the script reports a recommended confidence threshold based on maximum F1 score (`Best confidence threshold: 0.30`). You can re-run the script with this value using `--confidence-threshold` to obtain final results.*

---

Let me know if you'd like this exported as a Markdown or integrated into a documentation generator.


## 📜 License

MIT License. For academic use, please cite our paper.
