# ✈️ Mini CFD Post Processor

A simple post-processing tool for small CFD datasets made for quick aerodynamic visualization and analysis.  
Built using **Python**, **NumPy**, **Matplotlib**, and **Streamlit**, this tool lets you visualize velocity, vorticity, and pressure coefficient fields without needing heavy CFD software.

---

## 🧠 What it Does
- Reads `.npz` or `.csv` CFD output files  
- Plots:
  - **Vorticity contours**
  - **Velocity magnitude**
  - **Streamlines and quiver plots**
- Computes:
  - **Pressure coefficient (Cp)**
  - **Lift estimation from Cp**
- Runs both from:
  - **CLI (command line)** and  
  - **Web app (Streamlit dashboard)**  

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/japkaran12/mini-cfd-post.git
cd mini-cfd-post

### 2️⃣ Create virtual environment

python -m venv .venv
.\.venv\Scripts\activate

### 3️⃣ Install dependencies

pip install -r requirements.txt

🚀 How to Run
▶ Run the Streamlit App
python -m streamlit run streamlit_app.py

Then open in browser:
👉 http://localhost:8501

For local network access:
python -m streamlit run streamlit_app.py --server.address 0.0.0.0

🧩 Run from Command Line (CLI)
python -m mcp.cli info
python -m mcp.cli plot-vort examples/test_vort.npz --out vort_plot.png
python -m mcp.cli plot-vel examples/test_vort.npz --out vel_plot.png
python -m mcp.cli save-cp examples/test_vort.npz --out cp_data.csv

📁 Project Structure

mini-cfd-post/
│
├─ mcp/
│  ├─ io.py
│  ├─ fields.py
│  ├─ plotting.py
│  ├─ analysis.py
│  └─ cli.py
│
├─ examples/
│  ├─ test_vort.npz
│
├─ streamlit_app.py
├─ requirements.txt
└─ README.md

📚 Dependencies
streamlit
numpy
pandas
matplotlib
scipy

🧑‍💻 Author

Japkaran Singh Arneja
Aerospace Engineering, Lovely Professional University
Built as part of academic + personal research project for aerodynamic data visualization.

📜 License

MIT License — free to use and modify.