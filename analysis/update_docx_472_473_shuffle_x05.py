from __future__ import annotations

import copy
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "Draft_Mohammad_Taris_Syahir_Zul_Fahmi_6025242008.docx"
ANALYSIS = ROOT / "analysis"

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


TABLE_414 = [
    {
        "model": "EfficientNetV2M",
        "fp32_acc": "100,00%",
        "int8_acc": "99,76%",
        "fp32_size": "215,348",
        "int8_size": "58,473",
        "fp32_lat": "178,04",
        "int8_lat": "68,13",
        "size_fp32_v": 215.348,
        "size_int8_v": 58.473,
        "lat_fp32_v": 178.04,
        "lat_int8_v": 68.13,
    },
    {
        "model": "MobileNetV3Large",
        "fp32_acc": "99,88%",
        "int8_acc": "98,68%",
        "fp32_size": "21,077",
        "int8_size": "5,798",
        "fp32_lat": "15,49",
        "int8_lat": "8,39",
        "size_fp32_v": 21.077,
        "size_int8_v": 5.798,
        "lat_fp32_v": 15.49,
        "lat_int8_v": 8.39,
    },
    {
        "model": "EfficientNetLite0",
        "fp32_acc": "99,88%",
        "int8_acc": "99,76%",
        "fp32_size": "17,780",
        "int8_size": "5,008",
        "fp32_lat": "25,01",
        "int8_lat": "14,57",
        "size_fp32_v": 17.780,
        "size_int8_v": 5.008,
        "lat_fp32_v": 25.01,
        "lat_int8_v": 14.57,
    },
    {
        "model": "MobileNetV3Small",
        "fp32_acc": "99,64%",
        "int8_acc": "99,40%",
        "fp32_size": "9,505",
        "int8_size": "2,710",
        "fp32_lat": "6,31",
        "int8_lat": "5,09",
        "size_fp32_v": 9.505,
        "size_int8_v": 2.710,
        "lat_fp32_v": 6.31,
        "lat_int8_v": 5.09,
    },
    {
        "model": "ShuffleNetV2_x1_0",
        "fp32_acc": "99,16%",
        "int8_acc": "99,52%",
        "fp32_size": "8,528",
        "int8_size": "2,503",
        "fp32_lat": "6,85",
        "int8_lat": "5,58",
        "size_fp32_v": 8.528,
        "size_int8_v": 2.503,
        "lat_fp32_v": 6.85,
        "lat_int8_v": 5.58,
    },
    {
        "model": "ShuffleNetV2_x0_5",
        "fp32_acc": "98,20%",
        "int8_acc": "95,56%",
        "fp32_size": "4,897",
        "int8_size": "1,528",
        "fp32_lat": "3,75",
        "int8_lat": "3,25",
        "size_fp32_v": 4.897,
        "size_int8_v": 1.528,
        "lat_fp32_v": 3.75,
        "lat_int8_v": 3.25,
    },
    {
        "model": "NAS L0.05 C12 cells10 + KD + PTQ INT8",
        "fp32_acc": "99,76%",
        "int8_acc": "99,64%",
        "fp32_size": "2,440",
        "int8_size": "0,928",
        "fp32_lat": "4,79",
        "int8_lat": "3,87",
        "size_fp32_v": 2.440,
        "size_int8_v": 0.928,
        "lat_fp32_v": 4.79,
        "lat_int8_v": 3.87,
    },
]


def draw_grouped_log_bar(filename_stem: str, metric: str) -> None:
    models = [r["model"] for r in TABLE_414]
    y = np.arange(len(models))
    h = 0.34
    if metric == "size":
        fp32 = [r["size_fp32_v"] for r in TABLE_414]
        int8 = [r["size_int8_v"] for r in TABLE_414]
        fp32_labels = [f"{r['fp32_size']} MB" for r in TABLE_414]
        int8_labels = [f"{r['int8_size']} MB" for r in TABLE_414]
        title = "Perbandingan Ukuran ONNX FP32 dan ONNX INT8"
        xlabel = "Ukuran model (MB, skala log)"
        xlim = (0.45, 330)
    else:
        fp32 = [r["lat_fp32_v"] for r in TABLE_414]
        int8 = [r["lat_int8_v"] for r in TABLE_414]
        fp32_labels = [f"{r['fp32_lat']} ms" for r in TABLE_414]
        int8_labels = [f"{r['int8_lat']} ms" for r in TABLE_414]
        title = "Perbandingan Latency FP32 dan INT8 pada Raspberry Pi"
        xlabel = "Rerata latency (ms, skala log)"
        xlim = (0.75, 260)

    fig, ax = plt.subplots(figsize=(13.5, 7.2), dpi=180)
    ax.barh(y - h / 2, fp32, height=h, color="#2f6fb2", label="FP32")
    ax.barh(y + h / 2, int8, height=h, color="#15936b", label="INT8")
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=15, weight="bold", pad=14)
    ax.grid(axis="x", which="major", color="#d8dde6", linewidth=0.8, alpha=0.9)
    ax.grid(axis="x", which="minor", color="#edf0f5", linewidth=0.5, alpha=0.7)
    ax.legend(loc="lower right", frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for values, labels, offset in ((fp32, fp32_labels, -h / 2), (int8, int8_labels, h / 2)):
        for yy, val, label in zip(y, values, labels):
            ax.text(val * 1.05, yy + offset, label, va="center", ha="left", fontsize=8.5)

    fig.tight_layout()
    for ext in ("png", "svg"):
        out = ANALYSIS / f"{filename_stem}.{ext}"
        fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def w_tag(name: str) -> str:
    return f"{{{NS['w']}}}{name}"


def set_cell_text(tc: etree._Element, value: str) -> None:
    texts = tc.xpath(".//w:t", namespaces=NS)
    if not texts:
        p = etree.SubElement(tc, w_tag("p"))
        r = etree.SubElement(p, w_tag("r"))
        t = etree.SubElement(r, w_tag("t"))
        t.text = value
        return
    texts[0].text = value
    if value.startswith(" ") or value.endswith(" "):
        texts[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    for extra in texts[1:]:
        extra.text = ""


def replace_table_414(body: etree._Element) -> None:
    caption_idx = None
    for i, child in enumerate(body):
        txt = "".join(child.xpath(".//w:t/text()", namespaces=NS))
        if txt.strip() == "Tabel 4.14 Perbandingan FP32 dan INT8":
            caption_idx = i
            break
    if caption_idx is None:
        raise RuntimeError("Tabel 4.14 caption not found")

    tbl = None
    for child in body[caption_idx + 1 : caption_idx + 5]:
        if etree.QName(child).localname == "tbl":
            tbl = child
            break
    if tbl is None:
        raise RuntimeError("Tabel 4.14 table not found")

    rows = tbl.findall("w:tr", NS)
    header = rows[0]
    template = rows[1]
    for row in rows[1:]:
        tbl.remove(row)

    columns = ["model", "fp32_acc", "int8_acc", "fp32_size", "int8_size", "fp32_lat", "int8_lat"]
    for row_data in TABLE_414:
        row = copy.deepcopy(template)
        cells = row.findall("w:tc", NS)
        for cell, key in zip(cells, columns):
            set_cell_text(cell, row_data[key])
        tbl.append(row)


def clear_paragraph_content(p: etree._Element) -> None:
    for child in list(p):
        if etree.QName(child).localname != "pPr":
            p.remove(child)


def append_text_run(p: etree._Element, text: str, italic: bool = False) -> None:
    if not text:
        return
    r = etree.SubElement(p, w_tag("r"))
    if italic:
        rpr = etree.SubElement(r, w_tag("rPr"))
        etree.SubElement(rpr, w_tag("i"))
        etree.SubElement(rpr, w_tag("iCs"))
    t = etree.SubElement(r, w_tag("t"))
    if text.startswith(" ") or text.endswith(" "):
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text


def set_paragraph_marked(p: etree._Element, marked: str) -> None:
    clear_paragraph_content(p)
    parts = marked.split("*")
    for i, part in enumerate(parts):
        append_text_run(p, part, italic=(i % 2 == 1))


def clone_paragraph_with_text(reference_p: etree._Element, marked: str) -> etree._Element:
    p = etree.Element(w_tag("p"))
    ppr = reference_p.find("w:pPr", NS)
    if ppr is not None:
        p.append(copy.deepcopy(ppr))
    set_paragraph_marked(p, marked)
    return p


def replace_exact_paragraph(body: etree._Element, startswith: str, marked: str) -> int:
    for i, child in enumerate(body):
        if etree.QName(child).localname != "p":
            continue
        txt = "".join(child.xpath(".//w:t/text()", namespaces=NS))
        if txt.startswith(startswith):
            set_paragraph_marked(child, marked)
            return i
    raise RuntimeError(f"Paragraph not found: {startswith}")


def replace_473_paragraphs(body: etree._Element) -> None:
    heading_idx = None
    for i, child in enumerate(body):
        txt = "".join(child.xpath(".//w:t/text()", namespaces=NS))
        if txt.strip() == "4.7.3. Analisis Dampak PTQ INT8":
            heading_idx = i
            break
    if heading_idx is None:
        raise RuntimeError("4.7.3 heading not found")

    first_idx = heading_idx + 1
    ref = body[first_idx]
    new_texts = [
        "Dampak utama PTQ pada model NAS final adalah penurunan ukuran model dan *latency* dengan penurunan akurasi yang kecil. Model NAS L0.05 C12 cells10 + KD + PTQ INT8 mencapai akurasi 99,64%, ukuran 0,928 MB, dan rerata *latency* 3,87 ms pada Raspberry Pi. Dibandingkan ONNX FP32, akurasi turun 0,12 pp, tetapi ukuran model berkurang sekitar 2,63 kali dan *latency* menjadi lebih rendah.",
        "Hasil pada model pembanding menunjukkan bahwa ketahanan terhadap PTQ bersifat bergantung pada arsitektur. ShuffleNetV2_x0_5 memiliki jumlah parameter dan ukuran yang kecil, tetapi akurasi INT8 turun 2,64 pp. MobileNetV3Large dan EfficientNetLite0 juga menunjukkan penurunan akurasi setelah INT8, sedangkan ShuffleNetV2_x1_0 justru mengalami sedikit kenaikan akurasi pada hasil INT8. Oleh karena itu, evaluasi PTQ tidak cukup dibaca dari ukuran model atau FLOPs saja, tetapi perlu diuji langsung pada perangkat target.",
        "Dalam konteks penelitian ini, model NAS final tidak selalu menjadi model dengan *latency* absolut paling rendah, tetapi memberikan kompromi yang lebih seimbang antara akurasi, ukuran model, dan *latency*. Dibandingkan MobileNetV3Small dan ShuffleNetV2_x1_0, model NAS final memiliki ukuran INT8 lebih kecil dan *latency* lebih rendah. Dibandingkan ShuffleNetV2_x0_5, model NAS final sedikit lebih lambat, tetapi akurasinya jauh lebih tinggi. Temuan ini mendukung pemilihan NAS L0.05 C12 cells10 + KD + PTQ INT8 sebagai kandidat utama untuk evaluasi *deployment* pada Raspberry Pi.",
    ]
    set_paragraph_marked(ref, new_texts[0])
    insert_at = first_idx + 1
    for text in new_texts[1:]:
        body.insert(insert_at, clone_paragraph_with_text(ref, text))
        insert_at += 1


def update_paragraphs(body: etree._Element) -> None:
    replace_exact_paragraph(
        body,
        "Perbandingan performa model FP32 dan INT8 disajikan pada Tabel 4.14.",
        "Perbandingan performa model FP32 dan INT8 disajikan pada Tabel 4.14. Tabel ini memperlihatkan dampak PTQ terhadap akurasi, ukuran model, dan *latency* pada model *teacher*, model CNN *lightweight*, model super-*lightweight*, serta model NAS final. Berbeda dari Tabel 4.7 yang menggunakan ukuran *checkpoint* PyTorch, kolom ukuran pada Tabel 4.14 mengacu pada ukuran file ONNX sesuai format model yang diuji.",
    )
    replace_exact_paragraph(
        body,
        "Sebagaimana ditunjukkan pada Tabel 4.14, PTQ INT8 menurunkan ukuran model",
        "Sebagaimana ditunjukkan pada Tabel 4.14, PTQ INT8 menurunkan ukuran model pada seluruh model yang diuji, tetapi pengaruhnya terhadap akurasi tidak selalu sama. Pada model NAS L0.05 C12 cells10 + KD, ukuran model turun dari 2,440 MB pada ONNX FP32 menjadi 0,928 MB pada ONNX INT8, sementara akurasi INT8 tetap mencapai 99,64%. Sebaliknya, ShuffleNetV2_x0_5 memang memiliki *latency* INT8 yang rendah, yaitu 3,25 ms, tetapi akurasinya turun dari 98,20% menjadi 95,56%. Hasil ini menunjukkan bahwa ukuran kecil dan *latency* rendah belum cukup untuk menjamin stabilitas akurasi setelah PTQ.",
    )
    replace_exact_paragraph(
        body,
        "Gambar 4.11 memperlihatkan",
        "Gambar 4.11 memperlihatkan perubahan ukuran model setelah konversi INT8. Visualisasi ini menegaskan bahwa model NAS final memiliki ukuran INT8 paling kecil di antara model utama yang tetap mempertahankan akurasi tinggi. Selain ukuran model, dampak PTQ terhadap *latency* ditunjukkan pada Gambar 4.12.",
    )
    replace_exact_paragraph(
        body,
        "Gambar 4.12 memperlihatkan",
        "Gambar 4.12 memperlihatkan perubahan *latency* setelah konversi INT8. Pada model NAS final, *latency* turun dari 4,79 ms menjadi 3,87 ms. Walaupun ShuffleNetV2_x0_5 memiliki *latency* INT8 lebih rendah, penurunan akurasinya jauh lebih besar, sehingga model tersebut lebih tepat diposisikan sebagai pembanding super-*lightweight*, bukan kandidat akhir.",
    )
    replace_473_paragraphs(body)


def target_for_rid(rels_xml: bytes, rid: str) -> str:
    rels_root = etree.fromstring(rels_xml)
    for rel in rels_root:
        if rel.get("Id") == rid:
            return rel.get("Target")
    raise RuntimeError(f"Relationship {rid} not found")


def main() -> None:
    ANALYSIS.mkdir(exist_ok=True)
    draw_grouped_log_bar("fp32_int8_model_size_tabel414", "size")
    draw_grouped_log_bar("fp32_int8_latency_tabel414", "latency")

    backup = DOCX.with_name(
        f"{DOCX.stem}_backup_before_472_473_shuffle_x05_{datetime.now().strftime('%Y%m%d_%H%M%S')}{DOCX.suffix}"
    )
    shutil.copy2(DOCX, backup)

    with zipfile.ZipFile(DOCX, "r") as zin:
        files = {name: zin.read(name) for name in zin.namelist()}

    root = etree.fromstring(files["word/document.xml"])
    body = root.find("w:body", NS)
    if body is None:
        raise RuntimeError("document body not found")

    replace_table_414(body)
    update_paragraphs(body)
    files["word/document.xml"] = etree.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone="yes"
    )

    # Preserve existing relationship IDs and media filenames.
    image36_target = target_for_rid(files["word/_rels/document.xml.rels"], "rId49")
    image38_target = target_for_rid(files["word/_rels/document.xml.rels"], "rId51")
    files[f"word/{image36_target}"] = (ANALYSIS / "fp32_int8_model_size_tabel414.png").read_bytes()
    files[f"word/{image38_target}"] = (ANALYSIS / "fp32_int8_latency_tabel414.png").read_bytes()

    tmp = DOCX.with_suffix(".tmp.docx")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name, data in files.items():
            zout.writestr(name, data)
    tmp.replace(DOCX)

    print(f"backup={backup}")
    print("updated_docx=", DOCX)
    print("fig_size=", ANALYSIS / "fp32_int8_model_size_tabel414.png")
    print("fig_latency=", ANALYSIS / "fp32_int8_latency_tabel414.png")


if __name__ == "__main__":
    main()
