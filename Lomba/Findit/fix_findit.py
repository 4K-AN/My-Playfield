import json

with open("e:/GIT/My-Playfield/Lomba/Findit/FindIT.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell baru pengganti semua sel Colab/Kaggle download
local_setup_cell = {
    "cell_type": "code",
    "source": [
        "# ============================================================\n",
        "# DATASET LOKAL — sudah ada di ./dataset/\n",
        "# Struktur folder yang diharapkan:\n",
        "#   ./dataset/train/realperson/\n",
        "#   ./dataset/train/fake_printed/\n",
        "#   ./dataset/train/fake_screen/\n",
        "#   ./dataset/train/fake_mask/\n",
        "#   ./dataset/train/fake_mannequin/\n",
        "#   ./dataset/train/fake_unknown/\n",
        "#   ./dataset/test/\n",
        "# ============================================================\n",
        "import os\n",
        "\n",
        "TRAIN_PATH = './dataset/train'\n",
        "TEST_PATH  = './dataset/test'\n",
        "SAMPLE_SUB = './dataset/samplesubmission.csv'\n",
        "\n",
        "# Verifikasi dataset tersedia\n",
        "if os.path.exists(TRAIN_PATH):\n",
        "    print('✓ Dataset ditemukan!')\n",
        "    print(f'  Train : {os.path.abspath(TRAIN_PATH)}')\n",
        "    print(f'  Test  : {os.path.abspath(TEST_PATH)}')\n",
        "    print(f'\\n=== Folder Train ===')\n",
        "    for item in sorted(os.listdir(TRAIN_PATH)):\n",
        "        folder = os.path.join(TRAIN_PATH, item)\n",
        "        if os.path.isdir(folder):\n",
        "            count = len([f for f in os.listdir(folder)])\n",
        "            print(f'  {item}: {count} gambar')\n",
        "    print(f'\\n=== Folder Test ===')\n",
        "    print(f'  {len(os.listdir(TEST_PATH))} gambar test')\n",
        "else:\n",
        "    print('ERROR: Folder ./dataset/train tidak ditemukan!')\n",
        "    print('Pastikan dataset sudah diekstrak ke ./dataset/')\n"
    ],
    "metadata": {"id": "local_dataset_setup"},
    "execution_count": None,
    "outputs": []
}

# ID-ID sel yang perlu dihapus (semua sel Colab/Kaggle)
IDS_TO_REMOVE = {
    "WnWZ1XVdL-eU",   # mkdir kaggle
    "1Kgr33WlL_Qd",   # drive.mount
    "R8NsOiT5NXkW",   # kaggle download
    "Sl7-usmROLhn",   # unzip
}

new_cells = []
inserted = False

for cell in nb["cells"]:
    cell_id = cell.get("metadata", {}).get("id", "")

    # Hapus sel Colab/Kaggle
    if cell_id in IDS_TO_REMOVE:
        continue

    # Sisipkan sel lokal tepat sebelum sel "Cek sample submission"
    if cell_id == "df5zfxEjOZ4T" and not inserted:
        new_cells.append(local_setup_cell)
        inserted = True

    # Perbaiki sel yang masih pakai path /content/ menjadi ./dataset/
    source = "".join(cell.get("source", []))
    if "./dataset/" not in source and "/content/" in source:
        # Ganti reference Colab
        new_source = source.replace("/content/dataset/", "./dataset/")
        new_source = new_source.replace("/content/", "./")
        cell["source"] = [line + '\n' for line in new_source.split('\n')]
        if cell["source"]:
            cell["source"][-1] = cell["source"][-1].rstrip('\n')

    # Clear output lama agar tidak membingungkan
    if "outputs" in cell:
        cell["outputs"] = []
    if "execution_count" in cell:
        cell["execution_count"] = None

    new_cells.append(cell)

nb["cells"] = new_cells

# Update metadata agar tidak lagi tampak sebagai Colab
nb["metadata"].pop("colab", None)
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3 (FindIT DAC)",
    "language": "python",
    "name": "codegeex-agent"
}

with open("e:/GIT/My-Playfield/Lomba/Findit/FindIT.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"FindIT.ipynb berhasil diperbarui! Total cells: {len(new_cells)}")
