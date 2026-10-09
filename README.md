# Melih Baykal — Computer Vision and Machine Learning

Computer Engineering student at **Yıldız Technical University**, based in Istanbul. **Expected graduation: June 2027 · GPA: 3.39/4.00.**

My main focus is computer vision and deep learning: object detection, image classification, visual localization and PyTorch model development. This repository presents selected project case studies, the role I held in each, and separately attributed learning companions with runnable code.

[LinkedIn](https://www.linkedin.com/in/melih-baykal) · [Contact](mailto:melih.baykal@std.yildiz.edu.tr)

## TEKNOFEST Havacılıkta Yapay Zeka

**VEGA Team Captain · Finalist · 2026 · Completed**

I led a multidisciplinary team across all three competition tasks. The implementation details below describe our **team solution**.

Our TEKNOFEST team used Roboflow for image annotation.

| Task | Problem | Technical approach |
| --- | --- | --- |
| Object detection | Detect people and vehicles in RGB/thermal imagery; assess landing-zone suitability | Grounding DINO, tracking and landing-zone assessment |
| Position estimation | Estimate 3D position during GPS outages | DPVO visual odometry combined with MAT altitude estimates |
| Reference object detection | Locate previously unseen reference objects | Candidate masks, visual similarity and temporal reranking |

[Competition specification](https://cdn.t3kys.com/media/uploads/2026/09/15/leuhL1QEivLoMjz0AVIxAuacpB1vZS3n.pdf)

<details>
<summary>Recorded team evaluation: the false-positive / missed-person trade-off</summary>

This case study summarizes **VEGA team records dated 12 September 2026**. My confirmed role was team captain; implementation and measurements are attributed to the team. This documentation was prepared from existing reports: **the models were not rerun for this portfolio**.

The problem was to reject false person detections without discarding too many correct boxes. Grounding DINO candidates and tracking supplied ten-observation windows to an R3D18 video verifier; uncertain or incomplete windows remained unchanged.

The recorded **RGB offline evaluation** covered 10,235 frames: one 2021 competition video and 17 VisDrone VID test-dev sequences. It contained 170,491 candidate boxes and 6,648 tracks, not independent people. Matching used IoU ≥ 0.5. The 23,689 ignored/ambiguous candidates were excluded from TP/FP denominators.

| Recorded verifier | False-positive boxes removed / baseline FP | Correct boxes removed / baseline TP |
| --- | ---: | ---: |
| Earlier R3D18 | 6,432 / 85,837 (7.493%) | 12,358 / 60,965 (20.271%) |
| Selected R3D18 | 5,858 / 85,837 (6.825%) | 286 / 60,965 (0.469%) |

The selected configuration sacrificed some false-positive removal to preserve substantially more correct boxes. Reports record that model and thresholds were fixed before the long test. These percentages are box-filtering measures, **not mAP gains or overall recall**.

The aggregate loss hid a limitation: occluded-person boxes lost **215 / 21,318 (1.009%)**. Small, occluded and stationary subgroups overlap; their counts must not be added together.

On the recorded GTX 1660 Ti run, verification took **80.90 minutes**; detection/tracking plus verification took **192.42 minutes**. This was an offline second pass waiting for the tenth observation, **not real-time flight inference**. These historical team measurements are not an official competition score or a guarantee on new videos.

</details>

## MOSAIC image classification research

**Undergraduate Researcher · August–December 2025 · Completed**

- Prepared mammography images through DICOM conversion, normalization and filtering.
- Compared ConvNeXt and Swin Transformer architectures across CC/MLO views.
- Fine-tuned ResNet and MobileNet with learning-rate and optimizer choices.

**Focus:** preprocessing consistency, architecture comparison and model fine-tuning.

[MOSAIC research group](https://avesis.yildiz.edu.tr/arastirma-grubu/mosaic)

## Orphilume

**Film discovery and personalized recommendations · In development**

I am developing a film discovery platform combining keyword and semantic search, preference-driven ranking and recommendation explanations. Ratings, pairwise comparisons and imported viewing history support a user's taste profile.

**Project stack:** Python, FastAPI, React, PostgreSQL/pgvector, Meilisearch and Docker.

**Flow:** catalog ingestion → keyword and embedding search → preference-based ranking → explanations and feedback.

[Project website](https://orphilume.com)

## YTUHub

**Independent campus portal · In development**

I am developing an independent portal for Yıldız Technical University students, with events, saved calendars, project-team matching and a Gemini-powered campus assistant. Verified accounts, sessions and server-side persistence support access across devices.

**Project stack:** React, Node.js, Express, SQLite and Gemini API.

**Assistant scope:** published portal events and portal usage.

[Project website](https://ytuhub.com)

## Optimization algorithms from scratch

**Two-person course project · December 2024 · C and Python**

In a two-person course project, we implemented and compared Gradient Descent, SGD and Adam in C for binary Fashion-MNIST classification. Python visualizations supported inspection of training behavior and evaluation metrics.

**Focus:** translating mathematical update rules into executable training loops.

[Project report](https://online.yildiz.edu.tr/upload/ytu-test/Evaluation/23011505_MEL%C4%B0H_BAYKAL_e7a981fc-76db-4549-8dcd-c221084bfcb5.pdf)

The original coursework was jointly authored by **Ceyda Tolunay and Melih Baykal**; its original source files are currently unavailable.

[Runnable GD, SGD and Adam learning companion](demos/optimizer-comparison): a **new AI-assisted C/Python implementation created on 8 October 2026**, with validation-based learning-rate selection, held-out test evaluation, numerical checks and reproducible run artifacts. Its code and measured results are separate from the original 2024 project.

## Related experience

- **Koç University IT Intern, August–September 2026:** Gemini API applications, RAG, AI agents, load testing and technical research.
- **TEKNOFEST Sağlıkta Yapay Zeka Team Leader, 2025:** led a four-person team working on U-Net segmentation and CNN classification for MRI stroke detection.
- **YTÜ SKY LAB AIR LAB, 2024–present:** Video Understanding training, technical presentations and mentoring.

## Runnable computer vision evaluation

[OpenCV planar reference-matching learning companion](demos/reference-matching-evaluation) — **new AI-assisted educational example created on 8 October 2026**.

The CPU-only pipeline combines ORB descriptors, Hamming matching, a ratio test and RANSAC homography estimation. It includes deterministic synthetic scenes, geometric acceptance checks, separate development/test seeds, corner localization errors and an illustrated failure analysis.

On its 24 synthetic test cases, the fixed configuration detected 8 of 12 positive cases and rejected all 12 negative cases. Small blurred targets and one occluded target exposed failures. These measurements describe this small experiment; the code and results are separate from the VEGA competition work.

For optional installs using the package versions recorded in the educational demos, use Python 3.12.14 and a separate virtual environment for each companion. From the repository root, use `python -m pip install -r requirements-optimizer-recorded.txt` for [the optimizer snapshot](requirements-optimizer-recorded.txt), or `python -m pip install -r requirements-reference-recorded.txt` for [the reference-matching snapshot](requirements-reference-recorded.txt), then follow the corresponding demo's Run instructions. These snapshots pin top-level Python packages; transitive dependencies and the optimizer's compiler are not locked, and identical results across platforms are not guaranteed.

## Opportunities

Interested in computer vision and machine-learning internships or suitable early-career roles. I can start immediately and prefer 24 hours per week, subject to an agreed schedule.

During the university term, I must remain in Istanbul: I am available remotely Monday–Wednesday and on-site in Istanbul Thursday–Friday. On-site work in other cities or countries can only be considered outside the university term, with dates and arrangements agreed separately. Remote work from Istanbul with employers elsewhere can be considered subject to the applicable work eligibility and agreed terms.

---

The project technology lists describe the respective systems. Individual technical experience is summarized in my CV and can be discussed in more detail during an interview.