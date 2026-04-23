import markdown
from xhtml2pdf import pisa
import os

def convert_viva_to_pdf():
    input_filename = 'MINOR_PROJECT_VIVA.md'
    output_filename = 'MINOR_PROJECT_VIVA.pdf'

    with open(input_filename, 'r', encoding='utf-8') as f:
        md_text = f.read()

    html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])

    css = """
    <style>
        @page {
            size: a4 landscape;
            margin: 1.5cm;
        }
        body {
            font-family: 'Helvetica', Arial, sans-serif;
            font-size: 10pt;
            line-height: 1.4;
            color: #1a1a2e;
            background-color: #ffffff;
        }
        h1 {
            color: #0f3460;
            font-size: 18pt;
            border-bottom: 3px solid #e94560;
            padding-bottom: 6px;
            margin-top: 20px;
            display: block; clear: both;
        }
        h2 {
            color: #e94560;
            font-size: 15pt;
            border-bottom: 2px solid #0f3460;
            padding-bottom: 4px;
            margin-top: 20px;
            page-break-before: always;
            display: block; clear: both;
        }
        h3 {
            color: #0f3460;
            font-size: 12pt;
            margin-top: 14px;
            display: block; clear: both;
        }
        h4 {
            color: #533483;
            font-size: 11pt;
            margin-top: 10px;
            display: block; clear: both;
        }
        p {
            margin-bottom: 8px;
            text-align: justify;
            display: block;
        }
        ul {
            padding-left: 18px;
            margin-bottom: 10px;
        }
        li {
            padding-bottom: 4px;
            display: block;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0;
            font-size: 9pt;
        }
        th {
            background-color: #0f3460;
            color: #ffffff;
            padding: 6px 10px;
            text-align: left;
        }
        td {
            padding: 5px 10px;
            border-bottom: 1px solid #d0d0d0;
            vertical-align: top;
        }
        tr:nth-child(even) td {
            background-color: #f4f4f8;
        }
        code, pre {
            background-color: #f0f0f5;
            font-family: monospace;
            font-size: 8.5pt;
            padding: 2px 4px;
            display: block;
            white-space: pre-wrap;
        }
        blockquote {
            border-left: 4px solid #e94560;
            background-color: #fff5f7;
            padding: 8px 14px;
            color: #333;
            margin: 10px 0;
            display: block;
        }
        hr {
            border: 0;
            border-top: 1px solid #ccc;
            margin: 16px 0;
            display: block;
        }
        strong {
            color: #0f3460;
        }
    </style>
    """

    full_html = f"<html><head>{css}</head><body>{html_body}</body></html>"

    with open(output_filename, "w+b") as result_file:
        status = pisa.CreatePDF(full_html, dest=result_file)

    if status.err:
        print("Error during PDF generation.")
    else:
        print(f"Successfully generated {output_filename}")

if __name__ == "__main__":
    convert_viva_to_pdf()
