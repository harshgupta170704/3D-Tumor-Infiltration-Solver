with open('PINN_PAPERS_FOCUSED.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Publisher data for each paper — add after the Journal/Conference line
publishers = {
    # Format: unique string to find → publisher line to inject after it
    'arXiv:2105.09922':
        '- **Publisher:** Springer Nature (MICCAI proceedings via Lecture Notes in Computer Science)',

    '*PubMed / NeuroOncology* (PMID: 40899167)':
        '- **Journal:** *Neuro-Oncology Advances*\n- **Publisher:** Oxford University Press (OUP) on behalf of the Society for Neuro-Oncology',

    'PubMed* (PMID: 41850092)':
        '- **Publisher:** Elsevier (Computers in Biology and Medicine)',

    '*Scientific Reports*, 10, 11048':
        '- **Publisher:** Nature Portfolio / Springer Nature (open access)',

    '*Annals of Biomedical Engineering*, 49(4), 1295–1307':
        '- **Publisher:** Springer Nature on behalf of the Biomedical Engineering Society (BMES)',

    '*Medical Image Analysis*, 78, 102368':
        '- **Publisher:** Elsevier',

    'MICCAI 2023 Workshop on Data Augmentation, Labeling, and Imperfections':
        '- **Publisher:** Springer Nature (Lecture Notes in Computer Science, MICCAI workshop proceedings)',

    '*Frontiers in Neuroscience*, 16, 817808':
        '- **Publisher:** Frontiers Media S.A. (open access)',

    '*Medical Image Analysis*, 58, 101557':
        '- **Publisher:** Elsevier',

    '*IEEE Transactions on Medical Imaging* (Early Access)':
        '- **Publisher:** IEEE (Institute of Electrical and Electronics Engineers)',

    '*Medical Image Analysis*, 94, 103094':
        '- **Publisher:** Elsevier',
}

for marker, publisher_line in publishers.items():
    if marker in content:
        content = content.replace(marker, marker + '\n' + publisher_line)
        print(f"  Added publisher for: {marker[:55]}...")
    else:
        print(f"  NOT FOUND: {marker[:55]}...")

with open('PINN_PAPERS_FOCUSED.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("\nDone — publisher information added to all papers.")
