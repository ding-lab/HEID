from __future__ import annotations

COARSE11 = [
    "Tumor", "Fibroblast", "NK_T", "Myeloid_NOS", "NonMalignant_Parenchymal",
    "Endothelial", "Plasma", "SMC", "B_cell", "Schwann", "Others",
]
N11 = 11
SCORED10 = list(range(10))
OTHERS = 10


COARSE14 = [
    "Tumor", "Fibroblast", "NK_T", "Myeloid_NOS", "NonMalignant_Parenchymal",
    "Endothelial", "Plasma", "Pericyte/vSMC", "SMC", "B_cell", "Necrosis",
    "Mast", "Schwann", "Others",
]


FINE2REFINE = {
    "Tumor": "Tumor", "Tumor_ccRCC": "Tumor", "Tumor_pRCC": "Tumor",
    "Tumor_chRCC": "Tumor", "Neuroendocrine": "Tumor",
    "Fibroblast": "Fibroblast", "CAF": "Fibroblast",
    "NK_T": "NK_T", "T_cell": "NK_T", "NK": "NK_T",
    "Macrophage": "Macrophage", "Macrophage_Monocyte": "Macrophage",
    "Microglia": "Macrophage", "Kupffer": "Macrophage",
    "Intestinal_epithelium": "NonMalignant_Parenchymal", "Basal": "NonMalignant_Parenchymal",
    "Luminal": "NonMalignant_Parenchymal", "Hepatocyte": "NonMalignant_Parenchymal",
    "Acinar": "NonMalignant_Parenchymal", "Club": "NonMalignant_Parenchymal",
    "Duct_like": "NonMalignant_Parenchymal", "AT2": "NonMalignant_Parenchymal",
    "Epidermal": "NonMalignant_Parenchymal", "Dysplasia": "NonMalignant_Parenchymal",
    "Cholangiocyte": "NonMalignant_Parenchymal", "Basal/Myoepithelial": "NonMalignant_Parenchymal",
    "Squamous_cell": "NonMalignant_Parenchymal", "Islet": "NonMalignant_Parenchymal",
    "Normal_Luminal": "NonMalignant_Parenchymal", "Ciliated": "NonMalignant_Parenchymal",
    "Squamous_epithelium": "NonMalignant_Parenchymal", "AT1": "NonMalignant_Parenchymal",
    "IPMN": "NonMalignant_Parenchymal", "Basal_keratinocyte": "NonMalignant_Parenchymal",
    "Secretory_Epithelium": "NonMalignant_Parenchymal", "Astrocyte": "NonMalignant_Parenchymal",
    "NonMalignant_Parenchymal": "NonMalignant_Parenchymal",
    "Endothelial": "Endothelial", "Sinusoidal_endothelial": "Endothelial",
    "Plasma": "Plasma",
    "Pericyte/vSMC": "Pericyte/vSMC", "Pericyte": "Pericyte/vSMC", "vSMC": "Pericyte/vSMC",
    "Smooth_muscle": "SMC", "SMC": "SMC",
    "B_cell": "B_cell",
    "Unknown": "Others", "Oligodendrocyte": "Others", "Neuron": "Others",
    "Skeletal_muscle": "Others", "Adipocyte": "Others", "Others": "Others",
    "Necrosis": "Necrosis", "Necrotic": "Necrosis",
    "cDC2": "DC", "Langerhans_cell": "DC", "mregDC": "DC", "pDC": "DC",
    "cDC1": "DC", "cDC": "DC", "DC": "DC",
    "Neutrophil": "Neutrophil",
    "Mast": "Granulocyte", "Granulocyte": "Granulocyte",
    "Lymphatic_Endothelial": "Lymphatic_Endothelial",
    "Schwann": "Schwann",
}

REFINE2COARSE11 = {
    "Tumor": "Tumor", "Fibroblast": "Fibroblast", "NK_T": "NK_T",
    "Macrophage": "Myeloid_NOS", "DC": "Myeloid_NOS", "Neutrophil": "Myeloid_NOS",
    "NonMalignant_Parenchymal": "NonMalignant_Parenchymal",
    "Endothelial": "Endothelial", "Lymphatic_Endothelial": "Endothelial",
    "Plasma": "Plasma",
    "Pericyte/vSMC": "SMC", "SMC": "SMC",
    "B_cell": "B_cell", "Schwann": "Schwann",
    "Others": "Others",
    "Necrosis": None, "Granulocyte": "Others",
}
C11IDX = {c: i for i, c in enumerate(COARSE11)}
