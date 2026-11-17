# Model Card: Synthetic PneumoniaMNIST Generator (cDCGAN + Ray Serve)

> **Every output of this model is SYNTHETIC.** It is not patient data and not clinical evidence. Use it only for education and research prototyping.

| | |
|---|---|
| **Model** | Class-conditional DCGAN. The generator takes a label embedding concatenated with z, followed by ConvT-BN-ReLU layers and a Tanh output. The discriminator uses projection conditioning, with spectral norm optional. Output is 64x64 grayscale; the API can return 28x28. |
| **Version / id** | `artifacts/best_generator.pt`. The model id is the first 12 characters of the checkpoint's SHA-256 and is returned by `/health` and in every response. |
| **Selection** | Chosen by Ray Tune random search with ASHA (search space in report §5). The selection score was `val_F1_gain − 0.5·collapse − 0.5·memorisation`, where F1 gain is measured against a randomly-oversampled real baseline. |
| **Training data** | PneumoniaMNIST train split (4,708 images: 1,214 normal and 3,494 pneumonia). Images are 28x28 and were upsampled to 64x64. Source: Kermany et al. (2018), a paediatric (ages 1–5) chest X-ray cohort from a single centre in Guangzhou, China. The data is distributed via MedMNIST v2 under CC BY 4.0. |
| **Labels** | `0 = normal` and `1 = pneumonia`, taken from the MedMNIST INFO metadata. Pneumonia covers both bacterial and viral cases, which are not distinguished. |

## Intended use
* Teaching: showing what GAN training, conditioning, mode collapse and memorisation look like, and illustrating normal vs. pneumonia-like appearance at low resolution, always labelled as synthetic.
* Engineering: stress testing data pipelines, serving infrastructure, autoscaling and file-format handling.
* Methods research: studying how synthetic augmentation interacts with class imbalance, always compared against class-weighting and oversampling baselines.

## Out of scope (do not use for)
* Diagnosis, triage, screening, or any clinical decision support.
* Presenting outputs as real patient images, including in publications, slides, insurance or legal contexts.
* Creating or augmenting test sets, or making claims about diagnostic accuracy.
* Training models that will be deployed on patients without validation on real, external data.
* Adult populations, other modalities or other resolutions. The model has never seen these.

## Evaluation (filled from `outputs/eval/eval_summary.json`; see report §6–7)
* **Quality and conditioning:** class consistency is judged by a classifier trained on real data only. We report Fréchet distance per class in judge-feature space and the diversity ratio relative to real images.
* **Memorisation:** we compare each sample's nearest-neighbour pixel distance to the training set with the same distance for held-out real images. The flag rate for held-out real images is 5% by construction. We also inspect the closest pairs visually.
* **Utility:** we test downstream classifiers on the held-out test set across several seeds, with paired comparisons against real-only, class-weighted and oversampling+augmentation baselines. We also report a train-on-synthetic, test-on-real score.

## Known limitations and risks
* **Hallucination and label noise:** some samples generated for one class look like the other class to the judge (report Fig. 14). Synthetic labels are not ground truth.
* **Low fidelity:** the data is 28x28 in origin. Fine texture is smoothed, and clinically relevant detail is absent.
* **Coverage and bias:** diversity is below that of the real data (mode under-coverage), so rare presentations are under-represented. The training cohort is from a single site and a single age band, with no demographic metadata, so subgroup fairness cannot be assessed.
* **Privacy:** no near-copies were found by the pixel-space check. This is **not** a formal privacy guarantee: membership inference is still possible, and the model was not trained with differential privacy.
* **Evaluation pitfalls:** rebalancing alone (even with noise images) can raise macro-F1. The test split also has a different class prior from the training split. Small effects can be within seed variance.

## Misuse risks and mitigations
| Risk | Mitigation in this prototype | Still needed for any shared deployment |
|---|---|---|
| Fabricated "patient" images (fraud, fake papers) | `synthetic: true` and a disclaimer in every response; `X-Synthetic-Data` header; PNG tEXt provenance metadata | Robust invisible watermarking, auth, rate limits, audit logging |
| Mass generation / scraping | Cap of 64 images per request; bounded autoscaling (max 4 replicas) | Per-user quotas, API keys |
| Upload of real patient data | No upload or image-to-image endpoint; `extra="forbid"` rejects unknown fields such as `patient_id` | Data-governance review |
| Over-trust by learners | `/model_card` endpoint; disclaimers | Course materials explaining the limitations |

## Serving details
`POST /generate {label, n≤64, seed?, size∈{28,64}, return_grid?}`. The service runs on Ray Serve with 1–4 autoscaling replicas (target 4 ongoing requests per replica) and 0.25 GPU plus 1 CPU per replica. `@serve.batch` (up to 16 requests, 5 ms wait) merges concurrent requests. Outputs are deterministic for a given seed and do not depend on batching. Latency and throughput figures are in report §8.

**Contact / ownership:** DATA 255 course project. This is a prototype, not maintained for production.
