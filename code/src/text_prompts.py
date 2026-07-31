"""Dataset-level text prompts.

TNBC / MoNuSeg / GlaS prompts are the actual DeepSeek-R1 generated
morphological descriptions shown in this project's README.md (section 9a/9b/9c) —
copied verbatim, not the notebook's unused "{Prompt}" placeholder.

PanNuke has no DeepSeek-R1 description in this project (it isn't one of the
three datasets used in the paper); PANNUKE_PROMPT below is a short prompt
written by us for this project, clearly not LLM-generated, so it is not
mislabeled as a DeepSeek-R1 sample.
"""

TNBC_PROMPT = (
    "Based on the exceptionally detailed morphological account, which is highly characteristic of a "
    "high-grade invasive breast carcinoma, several critical diagnostic insights can be elaborated. The "
    "described high cellularity and marked nuclear pleomorphism encompassing both ovoid and spindled "
    "forms are hallmark features of a poorly differentiated neoplasm, often correlating with aggressive "
    "biological behavior and high histological grade (Grade 3). The alignment of these hyperchromatic, "
    "mitotically active nuclei along eosinophilic collagenous stroma is highly suggestive of an invasive "
    "carcinoma with a scirrhous or desmoplastic pattern, where the malignant cells induce a profound host "
    "stromal response. The prominent nuclear overlapping and concave junctions indicate a loss of cellular "
    "cohesion and a crowded, syncytial-like growth pattern, frequently seen in basal-like or triple-negative "
    "breast cancers."
)

MONUSEG_PROMPT = (
    "This detailed morphological account is quintessential for a high-grade invasive breast carcinoma, most "
    "likely of no special type (NST), and carries significant diagnostic and prognostic implications. The "
    "marked nuclear pleomorphism, encompassing both pyknotic and substantially enlarged forms with irregular "
    "outlines, is a cardinal feature of high histological grade (Grade 3, Nottingham grading system), "
    "directly correlating with increased genomic instability and aggressive tumor behavior. The observed "
    "chromatin pattern, hyperchromatic with clumping and prominent nucleoli, is highly suggestive of a high "
    "proliferative index and frequently associated with mutations in the TP53 tumor suppressor gene."
)

GLAS_PROMPT = (
    "Based on the described macroscopic morphology, this specimen exhibits a classic yet aggressive "
    "presentation of colorectal adenocarcinoma with several critical prognostic implications. The "
    "multifocal, multilobulated architecture with a dominant mass and true satellite nodules is highly "
    "characteristic of intramural metastatic spread, often via lymphovascular invasion, which significantly "
    "upstages the disease and increases the risk of local recurrence. The heterogeneous margin is a key "
    "diagnostic feature, with polypoid, sharply demarcated areas alongside spiculated, jagged invasion fronts."
)

PANNUKE_PROMPT = (
    "Multi-organ hematoxylin and eosin stained histology patch showing densely packed epithelial and stromal "
    "nuclei of varying size, shape and chromatin density, including neoplastic, inflammatory, connective, "
    "dead and normal epithelial nuclei populations against a heterogeneous tissue background."
)

DATASET_PROMPTS = {
    "TNBC": TNBC_PROMPT,
    "MoNuSeg": MONUSEG_PROMPT,
    "GlaS": GLAS_PROMPT,
    "PanNuke": PANNUKE_PROMPT,
}
