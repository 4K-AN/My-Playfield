import json

with open("e:/GIT/My-Playfield/Lomba/Findit/dac_find_it_2026.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# The site-packages path where pip successfully downloaded everything earlier
path_injector = (
    "import sys\n"
    "import os\n"
    "env_path = r'c:\\users\\admin\\.codegeex\\mamba\\envs\\codegeex-agent\\Lib\\site-packages'\n"
    "if os.path.exists(env_path) and env_path not in sys.path:\n"
    "    sys.path.insert(0, env_path)\n"
    "env_path2 = r'c:\\users\\admin\\.codegeex\\mamba\\envs\\codegeex-agent\\lib\\site-packages'\n"
    "if os.path.exists(env_path2) and env_path2 not in sys.path:\n"
    "    sys.path.insert(0, env_path2)\n\n"
)

# Find the cell containing the huge chunk of imports (e.g. pandas) and prepend the injector
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "import pandas as pd" in source and "sys.path.insert" not in source:
            # Reconstruct the source lines
            cell["source"] = [line + '\n' for line in (path_injector + source).split('\n') if line]

            # Let's also enforce it in Super Train cell at the bottom if any
        if "### super train restart ###" in source and "sys.path.insert" not in source:
            cell["source"] = [line + '\n' for line in (path_injector + source).split('\n') if line]

with open("e:/GIT/My-Playfield/Lomba/Findit/dac_find_it_2026.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
