import os
import urllib.request

papers_dir = r'C:\Users\harsh\OneDrive\Desktop\Cancer-Research\papers'
os.makedirs(papers_dir, exist_ok=True)

papers = [
    {
        'filename': 'Martens_2022_Deep_Learning_Glioma_Growth_Modeling.pdf',
        'url': 'https://arxiv.org/pdf/2111.13404.pdf',
        'title': 'Martens et al. (2022) - Deep Learning for Reaction-Diffusion Glioma Growth Modeling'
    },
    {
        'filename': 'Zhang_Lowengrub_2024_PINN_Glioblastoma_Infiltration.pdf',
        'url': 'https://arxiv.org/pdf/2311.16536.pdf',
        'title': 'Zhang, Lowengrub et al. (2024) - Personalized Predictions of Glioblastoma Infiltration (PINN)'
    },
    {
        'filename': 'Lipkova_Menze_2019_Personalized_Radiotherapy_Design.pdf',
        'url': 'https://arxiv.org/pdf/1807.00499.pdf',
        'title': 'Lipkova, Menze et al. (2019) - Personalized Radiotherapy Design for Glioblastoma'
    }
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
}

for p in papers:
    dest = os.path.join(papers_dir, p['filename'])
    print(f"Downloading: {p['title']}...")
    try:
        req = urllib.request.Request(p['url'], headers=headers)
        with urllib.request.urlopen(req) as response, open(dest, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  -> Successfully saved to {p['filename']} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"  -> Error downloading {p['filename']}: {e}")

print("\nAll downloads finished. Files in directory:")
for f in os.listdir(papers_dir):
    f_path = os.path.join(papers_dir, f)
    print(f"  - {f} ({os.path.getsize(f_path) / 1024 / 1024:.2f} MB)")
